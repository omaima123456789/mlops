from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import mlflow
import mlflow.sklearn

# ── Configuration MLflow ──────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Titanic-SVM")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API REST pour predire la survie sur le Titanic avec un modele SVM",
    version="1.0.0",
)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Modele introuvable : {MODEL_PATH}. Lancez 'make train' d'abord."
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
print(f"Modele charge depuis '{MODEL_PATH}'")


class PassengerData(BaseModel):
    pclass: int
    sex: str
    age: float
    fare: float

    class Config:
        json_schema_extra = {
            "example": {"pclass": 1, "sex": "female", "age": 29.0, "fare": 100.0}
        }


class RetrainData(BaseModel):
    kernel: str = "rbf"
    C: float = 1.0
    data_path: str = "titanic.csv"

    class Config:
        json_schema_extra = {
            "example": {"kernel": "rbf", "C": 1.0, "data_path": "titanic.csv"}
        }


@app.get("/")
def home():
    return {
        "message": "Bienvenue sur l'API Titanic !",
        "routes": {
            "prediction": "/predict",
            "retraining": "/retrain",
            "documentation": "/docs",
        },
    }


@app.post("/predict")
def predict(passenger: PassengerData):
    try:
        # CORRIGÉ : validation du sexe
        if passenger.sex not in ["male", "female"]:
            raise HTTPException(
                status_code=400, detail="Le sexe doit etre 'male' ou 'female'"
            )
        # CORRIGÉ : validation de la classe
        if passenger.pclass not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="pclass doit etre 1, 2 ou 3")
        # CORRIGÉ : validation de l'age
        if passenger.age <= 0 or passenger.age > 120:
            raise HTTPException(status_code=400, detail="age doit etre entre 0 et 120")

        sex_encoded = label_encoder.transform([passenger.sex])[0]
        features = np.array(
            [[passenger.pclass, sex_encoded, passenger.age, passenger.fare]]
        )
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        result = "Survivant" if prediction == 1 else "Non survivant"

        return {
            "prediction": int(prediction),
            "result": result,
            "details": {
                "pclass": passenger.pclass,
                "sex": passenger.sex,
                "age": passenger.age,
                "fare": passenger.fare,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prediction : {str(e)}")


@app.post("/retrain")
def retrain(params: RetrainData):
    try:
        # CORRIGÉ : validation du kernel
        if params.kernel not in ["rbf", "linear", "poly"]:
            raise HTTPException(
                status_code=400, detail="kernel doit etre 'rbf', 'linear' ou 'poly'"
            )

        if not os.path.exists(params.data_path):
            raise HTTPException(
                status_code=404, detail=f"Fichier introuvable : {params.data_path}"
            )

        from model_pipeline import prepare_data, train_model, evaluate_model, save_model

        # CORRIGÉ : run MLflow inclus dans le retrain (avant : MLflow non loggé)
        run_name = f"retrain_api_{params.kernel}_C{params.C}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("source", "api_retrain")
            mlflow.set_tag("model_type", "SVM")

            # CORRIGÉ : utilise train_model() du pipeline (avant : SVC() créé directement)
            X_train, X_test, y_train, y_test = prepare_data(params.data_path)
            new_model = train_model(X_train, y_train, kernel=params.kernel, C=params.C)
            score = evaluate_model(new_model, X_test, y_test)
            save_model(new_model, MODEL_PATH)

        global model
        model = joblib.load(MODEL_PATH)

        return {
            "message": "Modele reentraine avec succes !",
            "hyperparametres": {"kernel": params.kernel, "C": params.C},
            "accuracy": round(score, 4),
            "mlflow_run": run_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur de reentainement : {str(e)}"
        )
