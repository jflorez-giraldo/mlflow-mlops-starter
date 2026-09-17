"""Pipeline reproducible de entrenamiento, registro y promocion."""

import argparse
import json
import os
import subprocess

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from mlops_demo.config import DEFAULT_EXPERIMENT_NAME, model_name, tracking_uri
from mlops_demo.data import TARGET_COLUMN, load_iris_data, split_data
from mlops_demo.promote import PromotionResult, promote_candidate


def git_commit() -> str:
    value = os.getenv("GITHUB_SHA")
    if value:
        return value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unversioned"


def train_pipeline(
    *,
    tracking_server: str,
    registered_model_name: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    minimum_accuracy: float = 0.90,
    regularization: float = 1.0,
    random_state: int = 42,
) -> PromotionResult:
    mlflow.set_tracking_uri(tracking_server)
    mlflow.set_registry_uri(tracking_server)
    mlflow.set_experiment(experiment_name)

    features, target = load_iris_data()
    x_train, x_test, y_train, y_test = split_data(
        features,
        target,
        random_state=random_state,
    )
    params = {
        "C": regularization,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": random_state,
    }
    model = LogisticRegression(**params)

    with mlflow.start_run(run_name="iris-candidate") as run:
        training_dataset = x_train.copy()
        training_dataset[TARGET_COLUMN] = y_train
        dataset = mlflow.data.from_pandas(training_dataset, targets=TARGET_COLUMN, name="iris")
        mlflow.log_input(dataset, context="training")

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
        mlflow.set_tags(
            {
                "pipeline": "mlops-iris",
                "dataset": "sklearn-iris",
                "git.commit": git_commit(),
            }
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(x_train, model.predict(x_train)),
            input_example=x_train.head(3),
        )
        model_version = mlflow.register_model(model_info.model_uri, registered_model_name)
        mlflow.set_tag("registered_model.version", model_version.version)
        run_id = run.info.run_id

    result = promote_candidate(
        model_name=registered_model_name,
        version=model_version.version,
        accuracy=metrics["accuracy"],
        minimum_accuracy=minimum_accuracy,
        smoke_input=x_test.head(1),
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "model_name": registered_model_name,
                "model_version": result.version,
                "accuracy": metrics["accuracy"],
                "promoted": result.promoted,
                "reason": result.reason,
            },
            indent=2,
        )
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=tracking_uri())
    parser.add_argument("--model-name", default=model_name())
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--minimum-accuracy", type=float, default=0.90)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_pipeline(
        tracking_server=args.tracking_uri,
        registered_model_name=args.model_name,
        experiment_name=args.experiment_name,
        minimum_accuracy=args.minimum_accuracy,
        regularization=args.regularization,
        random_state=args.random_state,
    )
    if not result.promoted:
        raise SystemExit(f"El candidato no fue promovido: {result.reason}")


if __name__ == "__main__":
    main()
