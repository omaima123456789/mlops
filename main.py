import argparse
import mlflow
import mlflow.sklearn

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# ── Configuration MLflow avec backend SQLite (Excellence Atelier 5) ──
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Titanic-SVM")


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Titanic")
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "prepare", "train", "evaluate", "load"],
    )
    parser.add_argument("--data", default="titanic.csv")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 50)
    print(f"  ETAPE : {args.step.upper()}")
    print("=" * 50)

    # Nom du run lisible dans l'interface MLflow (Excellence)
    run_name = f"svm_{args.kernel}_C{args.C}"

    with mlflow.start_run(run_name=run_name):
        # Tags pour organiser les runs (Excellence)
        mlflow.set_tag("model_type", "SVM")
        mlflow.set_tag("dataset", "Titanic")
        mlflow.set_tag("step", args.step)

        X_train, X_test, y_train, y_test = prepare_data(args.data)

        if args.step in ["train", "all"]:
            print("\nEntrainement du modele...")
            model = train_model(X_train, y_train, kernel=args.kernel, C=args.C)
            score = evaluate_model(model, X_test, y_test)

            # Log MLflow ici uniquement (pas dans model_pipeline)
            mlflow.log_param("kernel", args.kernel)
            mlflow.log_param("C", args.C)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", score)
            mlflow.sklearn.log_model(model, "model")

            save_model(model, args.model)
            print(f"Accuracy : {score:.4f}")

        if args.step == "evaluate":
            model = load_model(args.model)
            score = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("accuracy", score)
            print(f"Accuracy : {score:.4f}")

        if args.step == "load":
            model = load_model(args.model)
            print(f"Modele charge : {type(model).__name__}")

    print("\n" + "=" * 50)
    print("  TERMINE AVEC SUCCES")
    print("  MLflow : http://127.0.0.1:5000")
    print("=" * 50)


if __name__ == "__main__":
    main()

# trigger CI
