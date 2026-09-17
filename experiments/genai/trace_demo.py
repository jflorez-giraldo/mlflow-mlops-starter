"""Genera una traza GenAI local sin depender de un proveedor externo."""

import os

import mlflow
from mlflow.entities import SpanType

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "genai-tracing-demo"


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_context(question: str) -> list[str]:
    """Simula la recuperacion de contexto para mantener el ejemplo sin credenciales."""
    documents = {
        "mlflow": "MLflow registra experimentos, modelos, evaluaciones y trazas.",
        "uv": "uv administra Python, entornos virtuales y dependencias reproducibles.",
    }
    lowered_question = question.lower()
    return [text for topic, text in documents.items() if topic in lowered_question]


@mlflow.trace(span_type=SpanType.LLM, attributes={"model": "local-rule-based-demo"})
def generate_answer(question: str, context: list[str]) -> str:
    """Simula la respuesta de un LLM para demostrar un span de tipo LLM."""
    if not context:
        return f"No encontre contexto local para responder: {question}"
    return " ".join(context)


@mlflow.trace(name="answer-question", span_type=SpanType.CHAIN)
def answer_question(question: str) -> str:
    mlflow.update_current_trace(
        tags={"environment": "local", "example": "manual-tracing"},
        metadata={"mlflow.trace.user": "demo-user", "mlflow.trace.session": "first-session"},
    )
    context = retrieve_context(question)
    return generate_answer(question, context)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    question = "Como se complementan MLflow y uv?"
    answer = answer_question(question)

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Pregunta: {question}")
    print(f"Respuesta: {answer}")
    print("Consulta la pestaña Traces del experimento genai-tracing-demo.")


if __name__ == "__main__":
    main()
