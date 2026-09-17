import pandas as pd
import pytest

from mlops_demo.data import FEATURE_COLUMNS, load_iris_data, split_data, validate_dataset


def test_iris_dataset_has_expected_contract() -> None:
    features, target = load_iris_data()

    assert features.shape == (150, 4)
    assert list(features.columns) == FEATURE_COLUMNS
    assert sorted(target.unique().tolist()) == [0, 1, 2]


def test_split_is_reproducible() -> None:
    features, target = load_iris_data()
    first = split_data(features, target, random_state=42)
    second = split_data(features, target, random_state=42)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_series_equal(first[3], second[3])


def test_validation_rejects_missing_values() -> None:
    features, target = load_iris_data()
    features.loc[0, FEATURE_COLUMNS[0]] = None

    with pytest.raises(ValueError, match="valores nulos"):
        validate_dataset(features, target)
