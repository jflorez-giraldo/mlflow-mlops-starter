"""Entrena, registra y vuelve a cargar un clasificador Iris con MLflow."""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "classic-ml-iris"


def main() -> None:
    iris = datasets.load_iris(as_frame=True)
    features = iris.data
    target = iris.target
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    params = {"solver": "lbfgs", "max_iter": 1000, "random_state": 42}
    model = LogisticRegression(**params)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logistic-regression") as run:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision_weighted": precision_score(y_test, predictions, average="weighted"),
            "recall_weighted": recall_score(y_test, predictions, average="weighted"),
            "f1_weighted": f1_score(y_test, predictions, average="weighted"),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("training.dataset", "sklearn-iris")

        signature = infer_signature(x_train, model.predict(x_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="iris_model",
            signature=signature,
            input_example=x_train.head(3),
        )

        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        sample = x_test.head(4)
        result = sample.copy()
        result["actual_class"] = y_test.loc[sample.index]
        result["predicted_class"] = loaded_model.predict(sample).astype(int)

        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")
        print(pd.DataFrame(metrics, index=["value"]).to_string())
        print("\nPredicciones de comprobacion:")
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
