"""API FastAPI que sirve la version champion registrada en MLflow."""

from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from mlflow import MlflowClient

from mlops_demo.config import model_name, tracking_uri
from mlops_demo.data import CLASS_NAMES, FEATURE_COLUMNS
from mlops_demo.schemas import Health, IrisFeatures, Prediction


def load_champion() -> tuple[Any, str, str]:
    uri = tracking_uri()
    name = model_name()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    version = MlflowClient().get_model_version_by_alias(name, "champion")
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    return model, name, str(version.version)


def prediction_frame(features: IrisFeatures) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                FEATURE_COLUMNS[0]: features.sepal_length_cm,
                FEATURE_COLUMNS[1]: features.sepal_width_cm,
                FEATURE_COLUMNS[2]: features.petal_length_cm,
                FEATURE_COLUMNS[3]: features.petal_width_cm,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def create_app(*, load_model_on_startup: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_model_on_startup:
            app.state.model, app.state.model_name, app.state.model_version = load_champion()
        yield

    application = FastAPI(
        title="Iris Classifier API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "Iris Classifier API",
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict",
        }

    @application.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        return Health(
            status="ok",
            model_name=request.app.state.model_name,
            model_version=request.app.state.model_version,
        )

    @application.post("/predict", response_model=Prediction)
    def predict(features: IrisFeatures, request: Request) -> Prediction:
        prediction = request.app.state.model.predict(prediction_frame(features))
        class_id = int(prediction[0])
        return Prediction(
            class_id=class_id,
            class_name=CLASS_NAMES[class_id],
            model_version=request.app.state.model_version,
        )

    return application


app = create_app()


def main() -> None:
    uvicorn.run("mlops_demo.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
