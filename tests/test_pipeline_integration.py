import mlflow
from mlflow import MlflowClient

from mlops_demo.data import load_iris_data
from mlops_demo.train import train_pipeline


def test_pipeline_registers_and_promotes_model(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow-test.db'}"
    model_name = "test-iris-classifier"

    result = train_pipeline(
        tracking_server=tracking_uri,
        registered_model_name=model_name,
        experiment_name="integration-test",
        minimum_accuracy=0.90,
    )

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    champion = client.get_model_version_by_alias(model_name, "champion")
    loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    features, _ = load_iris_data()

    assert result.promoted is True
    assert champion.version == result.version
    assert len(loaded_model.predict(features.head(2))) == 2

    rejected = train_pipeline(
        tracking_server=tracking_uri,
        registered_model_name=model_name,
        experiment_name="integration-test",
        minimum_accuracy=0.99,
    )
    current_champion = client.get_model_version_by_alias(model_name, "champion")
    candidate = client.get_model_version_by_alias(model_name, "candidate")

    assert rejected.promoted is False
    assert current_champion.version == result.version
    assert candidate.version == rejected.version
    assert candidate.tags["validation_status"] == "rejected"
