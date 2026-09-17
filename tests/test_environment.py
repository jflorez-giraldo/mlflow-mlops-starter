"""Comprobaciones rapidas del entorno y de las APIs usadas por los ejemplos."""

import unittest

import mlflow
import pandas
import sklearn
from mlflow.entities import SpanType


class EnvironmentTest(unittest.TestCase):
    def test_core_dependencies_are_importable(self) -> None:
        self.assertTrue(mlflow.__version__)
        self.assertTrue(pandas.__version__)
        self.assertTrue(sklearn.__version__)

    def test_tracing_api_is_available(self) -> None:
        self.assertTrue(callable(mlflow.trace))
        self.assertEqual(SpanType.CHAIN, "CHAIN")


if __name__ == "__main__":
    unittest.main()
