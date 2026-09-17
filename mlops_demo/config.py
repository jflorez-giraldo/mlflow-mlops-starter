"""Configuracion compartida por entrenamiento y serving."""

import os

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_MODEL_NAME = "iris-classifier"
DEFAULT_EXPERIMENT_NAME = "mlops-iris-training"


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def model_name() -> str:
    return os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
