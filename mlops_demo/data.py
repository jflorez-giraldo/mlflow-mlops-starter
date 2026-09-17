"""Carga y validacion del dataset usado por el pipeline."""

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
TARGET_COLUMN = "target"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_iris_data() -> tuple[pd.DataFrame, pd.Series]:
    iris = datasets.load_iris(as_frame=True)
    features = iris.data.copy()
    target = iris.target.copy()
    validate_dataset(features, target)
    return features, target


def validate_dataset(features: pd.DataFrame, target: pd.Series) -> None:
    if list(features.columns) != FEATURE_COLUMNS:
        raise ValueError(f"Columnas inesperadas: {list(features.columns)}")
    if features.empty or len(features) != len(target):
        raise ValueError("Features y target deben contener el mismo numero de filas")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("El dataset no puede contener valores nulos")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("Todas las features deben ser numericas")
    if sorted(target.unique().tolist()) != [0, 1, 2]:
        raise ValueError("El target debe contener exactamente las clases 0, 1 y 2")


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
