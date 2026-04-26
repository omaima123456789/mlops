import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import joblib
import os


def prepare_data(path):
    """
    Charge et prétraite les données du fichier CSV.

    Args:
        path (str): Chemin vers le fichier CSV

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if not path:
        raise ValueError("Le chemin du fichier ne peut pas être vide")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier '{path}' est introuvable")

    data = pd.read_csv(path)
    data = data[["Survived", "Pclass", "Sex", "Age", "Fare"]]
    data["Age"] = data["Age"].fillna(data["Age"].mean())
    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

    le = LabelEncoder()
    data["Sex"] = le.fit_transform(data["Sex"])
    joblib.dump(le, "label_encoder.pkl")

    X = data.drop("Survived", axis=1)
    y = data["Survived"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train, kernel="rbf", C=1.0):
    """
    Entraîne un modèle SVM.

    Args:
        X_train (array): Données d'entraînement
        y_train (array): Labels d'entraînement
        kernel (str): Noyau SVM (rbf, linear, poly)
        C (float): Paramètre de régularisation

    Returns:
        SVC: Modèle entraîné
    """
    # CORRIGÉ : ajout des validations nécessaires pour les tests
    if X_train is None or y_train is None:
        raise ValueError("Les données d'entraînement ne peuvent pas être None")
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Les données d'entraînement ne peuvent pas être vides")

    model = SVC(kernel=kernel, C=C, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle.

    Args:
        model (SVC): Modèle entraîné
        X_test (array): Données de test
        y_test (array): Labels de test

    Returns:
        float: Score d'accuracy
    """
    # CORRIGÉ : ajout des validations
    if model is None:
        raise ValueError("Le modèle ne peut pas être None")
    if X_test is None or y_test is None:
        raise ValueError("Les données de test ne peuvent pas être None")

    return model.score(X_test, y_test)


def save_model(model, filename):
    """
    Sauvegarde le modèle avec joblib.

    Args:
        model (SVC): Modèle à sauvegarder
        filename (str): Nom du fichier
    """
    # CORRIGÉ : ajout des validations
    if model is None:
        raise ValueError("Le modèle ne peut pas être None")
    if not filename:
        raise ValueError("Le nom du fichier ne peut pas être vide")

    joblib.dump(model, filename)
    print(f"Modèle sauvegardé dans '{filename}'")


def load_model(filename):
    """
    Charge un modèle sauvegardé depuis un fichier.

    Args:
        filename (str): Nom du fichier du modèle

    Returns:
        SVC: Modèle chargé
    """
    # CORRIGÉ : vérification filename vide ajoutée
    if not filename:
        raise ValueError("Le nom du fichier ne peut pas être vide")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Le fichier modèle '{filename}' est introuvable")

    return joblib.load(filename)
