"""Contratos HTTP de la API de inferencia."""

from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    sepal_length_cm: float = Field(gt=0)
    sepal_width_cm: float = Field(gt=0)
    petal_length_cm: float = Field(gt=0)
    petal_width_cm: float = Field(gt=0)


class Prediction(BaseModel):
    class_id: int
    class_name: str
    model_version: str


class Health(BaseModel):
    status: str
    model_name: str
    model_version: str
