"""Reglas de calidad y promocion de versiones en Model Registry."""

from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    reason: str
    version: int


def passes_quality_gate(
    candidate_accuracy: float,
    champion_accuracy: float | None,
    minimum_accuracy: float,
) -> tuple[bool, str]:
    if candidate_accuracy < minimum_accuracy:
        return False, f"accuracy {candidate_accuracy:.4f} < umbral {minimum_accuracy:.4f}"
    if champion_accuracy is not None and candidate_accuracy < champion_accuracy:
        return False, (f"accuracy {candidate_accuracy:.4f} < champion {champion_accuracy:.4f}")
    return True, "quality gate superado"


def promote_candidate(
    *,
    model_name: str,
    version: int | str,
    accuracy: float,
    minimum_accuracy: float,
    smoke_input: pd.DataFrame,
    client: MlflowClient | None = None,
) -> PromotionResult:
    registry = client or MlflowClient()
    registry.set_registered_model_alias(model_name, "candidate", version)
    registry.set_model_version_tag(model_name, version, "validation_accuracy", str(accuracy))

    champion_accuracy = None
    try:
        champion = registry.get_model_version_by_alias(model_name, "champion")
        stored_accuracy = champion.tags.get("validation_accuracy")
        champion_accuracy = float(stored_accuracy) if stored_accuracy is not None else None
    except MlflowException:
        pass

    accepted, reason = passes_quality_gate(accuracy, champion_accuracy, minimum_accuracy)
    if accepted:
        try:
            candidate_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
            prediction = candidate_model.predict(smoke_input)
            if len(prediction) != len(smoke_input):
                raise ValueError("La prueba de inferencia devolvio un numero inesperado de filas")
        except Exception as error:  # noqa: BLE001 - any model loading/inference failure rejects it
            accepted = False
            reason = f"smoke test fallido: {error}"

    status = "passed" if accepted else "rejected"
    registry.set_model_version_tag(model_name, version, "validation_status", status)
    registry.set_model_version_tag(model_name, version, "validation_reason", reason)
    if accepted:
        registry.set_registered_model_alias(model_name, "champion", version)

    return PromotionResult(promoted=accepted, reason=reason, version=int(version))
