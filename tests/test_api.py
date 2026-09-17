import numpy as np
from fastapi.testclient import TestClient

from mlops_demo.api import create_app, prediction_frame
from mlops_demo.data import FEATURE_COLUMNS
from mlops_demo.schemas import IrisFeatures


class FakeModel:
    def predict(self, data):
        assert list(data.columns) == FEATURE_COLUMNS
        return np.array([0])


def test_prediction_frame_preserves_training_schema() -> None:
    features = IrisFeatures(
        sepal_length_cm=5.1,
        sepal_width_cm=3.5,
        petal_length_cm=1.4,
        petal_width_cm=0.2,
    )
    frame = prediction_frame(features)

    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.shape == (1, 4)


def test_health_and_prediction_endpoints() -> None:
    app = create_app(load_model_on_startup=False)
    app.state.model = FakeModel()
    app.state.model_name = "iris-classifier"
    app.state.model_version = "7"

    with TestClient(app) as client:
        root = client.get("/")
        health = client.get("/health")
        response = client.post(
            "/predict",
            json={
                "sepal_length_cm": 5.1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            },
        )

    assert root.json() == {
        "name": "Iris Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
    assert health.json() == {
        "status": "ok",
        "model_name": "iris-classifier",
        "model_version": "7",
    }
    assert response.json() == {"class_id": 0, "class_name": "setosa", "model_version": "7"}


def test_prediction_rejects_invalid_measurements() -> None:
    app = create_app(load_model_on_startup=False)
    app.state.model = FakeModel()
    app.state.model_name = "iris-classifier"
    app.state.model_version = "1"

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "sepal_length_cm": -1,
                "sepal_width_cm": 3.5,
                "petal_length_cm": 1.4,
                "petal_width_cm": 0.2,
            },
        )

    assert response.status_code == 422
