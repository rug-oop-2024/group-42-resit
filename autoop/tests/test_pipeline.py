import unittest

import pandas as pd
from sklearn.datasets import fetch_openml

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Accuracy, MeanSquaredError
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types


class TestPipeline(unittest.TestCase):

    def setUp(self) -> None:
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=get_model("multiple_linear_regression"),
            input_features=list(filter(lambda x: x._name != "age",
                                       self.features)),
            target_feature=Feature(name="age", type="continuous"),
            metrics=[MeanSquaredError(), Accuracy()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self):
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self):
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(self.pipeline._train_X[0].shape[0], int(0.8 * self.ds_size))
        self.assertEqual(self.pipeline._test_X[0].shape[0],
                         self.ds_size - int(0.8 * self.ds_size))

    def test_train(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate()
        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._evaluation_metrics_results)
        self.assertEqual(len(self.pipeline._evaluation_metrics_results), 2)
