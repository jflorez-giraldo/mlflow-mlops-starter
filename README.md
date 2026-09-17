# MLflow Starter

[![CI](https://github.com/jflorez-giraldo/mlflow-mlops-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/jflorez-giraldo/mlflow-mlops-starter/actions/workflows/ci.yml)

Proyecto local reproducible para aprender MLflow 3 y recorrer un ciclo MLOps completo:

- Entrenamiento, evaluacion y registro de un modelo con scikit-learn.
- Promocion automatica mediante los alias `candidate` y `champion`.
- API FastAPI que sirve el champion registrado.
- Stack Docker Compose y validacion continua con GitHub Actions.
- Trazas de una aplicacion GenAI simulada, sin claves ni servicios externos.

La configuracion sigue los quickstarts oficiales de [MLflow Tracking](https://mlflow.org/docs/latest/ml/getting-started/quickstart/) y [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/quickstart/). Para desarrollo local usa SQLite, recomendado por MLflow frente al backend de archivos en modo de mantenimiento.

## Flujo MLOps

```text
Iris -> validacion -> entrenamiento -> metricas -> Model Registry
                                                   |
                                      quality gate + smoke test
                                                   |
                                      alias champion -> FastAPI
```

Cada entrenamiento crea una version de `iris-classifier` y mueve el alias `candidate`. La version solo recibe el alias `champion` cuando su accuracy es al menos `0.90`, no empeora el champion actual y supera una inferencia de prueba. Los stages tradicionales de MLflow no se usan porque estan obsoletos en favor de aliases y tags.

## Inicio rapido con Docker

Docker Compose mantiene MLflow, SQLite y los artefactos en volumenes persistentes:

```bash
docker compose up -d mlflow
docker compose run --rm trainer
docker compose up -d api
```

Servicios disponibles:

- MLflow UI: <http://127.0.0.1:5001>
- API: <http://127.0.0.1:8000>
- Swagger: <http://127.0.0.1:8000/docs>

La raiz de la API muestra un indice JSON con las rutas disponibles.

Comprobar la API:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length_cm": 5.1,
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2
  }'
```

Después de promover otro modelo se debe reiniciar la API para cargar el nuevo champion:

```bash
docker compose restart api
```

## Requisitos

- `uv` instalado.
- Un puerto local `5001` disponible (`5000` suele estar ocupado por AirPlay en macOS).

`uv` instala automáticamente Python 3.12 y crea `.venv` cuando sea necesario.

## Preparar el entorno

```bash
uv sync
```

No es necesario activar el entorno: `uv run` sincroniza `.venv` y ejecuta cada comando con las versiones de `uv.lock`. Si se desea activarlo manualmente en macOS o Linux:

```bash
source .venv/bin/activate
```

## Iniciar MLflow

Desde la raiz del proyecto:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5001
```

Abrir <http://127.0.0.1:5001>. La base `mlflow.db` guarda experimentos, runs y trazas; `mlartifacts/` contiene modelos y otros artefactos. Ambos son estado local ignorado por Git.

### Pipeline y API locales

Con MLflow activo:

```bash
uv run python -m mlops_demo.train
uv run python -m mlops_demo.api
```

Opciones del entrenamiento:

```bash
uv run python -m mlops_demo.train \
  --minimum-accuracy 0.90 \
  --regularization 1.0 \
  --random-state 42
```

También pueden configurarse `MLFLOW_TRACKING_URI` y `MLFLOW_MODEL_NAME`. La API carga `models:/iris-classifier@champion` una vez durante su arranque.

## Ejecutar los ejemplos

Con el servidor activo en otra terminal:

```bash
uv run python experiments/classic_ml/train_iris.py
uv run python experiments/genai/trace_demo.py
```

El primer comando crea el experimento `classic-ml-iris`, registra parametros, metricas y un modelo, y comprueba que el modelo registrado como artefacto puede volver a cargarse. El segundo crea `genai-tracing-demo` y una traza con spans padre-hijo visible en la pestaña **Traces**.

Puede apuntarse a otro servidor sin modificar código:

```bash
MLFLOW_TRACKING_URI=https://mi-servidor.example uv run python experiments/classic_ml/train_iris.py
```

## Notebooks

Iniciar Jupyter sin agregar todo Jupyter a las dependencias permanentes:

```bash
uv run --with jupyter jupyter lab
```

Seleccionar el interprete `.venv/bin/python` o el kernel Python del proyecto y abrir:

- `notebooks/01_classic_ml.ipynb`
- `notebooks/02_genai_tracing.ipynb`

`ipykernel` sí está bloqueado como dependencia de desarrollo para que VS Code y Jupyter puedan usar `.venv` directamente.

## Comprobaciones

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m pytest -q
```

Las pruebas incluyen contrato de datos, split reproducible, quality gate, endpoints y un pipeline integral con SQLite y Model Registry temporales.

## Integracion continua

`.github/workflows/ci.yml` ejecuta en cada push y pull request:

1. Sincronizacion exacta desde `uv.lock`.
2. Lint y comprobacion de formato con Ruff.
3. Pruebas unitarias e integrales.
4. Entrenamiento, promocion y smoke test HTTP de la API.
5. Construccion de la imagen Docker y, en `main`, publicacion en GHCR con etiquetas `latest` y SHA.

La imagen publicada queda disponible como:

```text
ghcr.io/jflorez-giraldo/mlflow-mlops-starter:latest
```

El repositorio esta publicado en [GitHub](https://github.com/jflorez-giraldo/mlflow-mlops-starter).
La rama `main` exige que los jobs `quality-and-integration` y `docker` terminen correctamente,
bloquea force-push y publica la imagen aprobada en
[GHCR](https://github.com/users/jflorez-giraldo/packages/container/package/mlflow-mlops-starter).

## `uv` frente a `uvx`

- `uv sync` crea y sincroniza el entorno del proyecto.
- `uv run ...` usa las dependencias exactas de `uv.lock`; es la opción normal en este repositorio.
- `uvx mlflow@latest ...` ejecuta una copia temporal y aislada de MLflow. Es útil para probar herramientas puntuales, pero no para el servidor habitual porque podría usar una versión distinta de la del proyecto.

## Conceptos que se observan

- **Experimento:** contenedor lógico de runs o trazas relacionadas.
- **Run:** ejecución de entrenamiento con parámetros, métricas y artefactos.
- **Modelo:** artefacto con sabores de MLflow para cargarlo o desplegarlo.
- **Trace:** recorrido de una solicitud por una aplicación GenAI.
- **Span:** operación individual dentro de una traza.

## Extender el ejemplo GenAI con OpenAI

El ejemplo inicial usa tracing manual y no necesita credenciales. Para probar el autologging del quickstart oficial:

```bash
uv add openai
export OPENAI_API_KEY="..."
```

Después se puede habilitar `mlflow.openai.autolog()` antes de invocar el cliente de OpenAI. No se deben guardar claves en el repositorio.
