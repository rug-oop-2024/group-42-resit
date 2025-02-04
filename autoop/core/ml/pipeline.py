import io
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric, get_metric
from autoop.core.ml.model import Model, get_model
from autoop.functional.feature import detect_feature_types_from_dataframe
from autoop.functional.preprocessing import preprocess_features


class Pipeline():
    """
    Pipline class, it takes data to train a model on
    and then can perform certain metrics on it.
    """

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """
        Initializes the pipline.
        Args:
            metrics[List[Metric]]: The list of metric to be performed
            dataset[Dataset]: The data to be used.
            model[Model]: The model to be used.
            input_features[List[Feature]]: The features
            (from the data) to be used.
            target_feature[Feature]: The target feature
            (the feature to test on).
            split[float]: The split between the training and testing data.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        self._training_metrics_results = []
        self._evaluation_metrics_results = []
        self._collapsed_y = False
        if target_feature.type == "categorical" and (
                model.type != "classification"):
            target_string = " classification for categorical target feature"
            raise ValueError("Model type must be" + target_string)
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature")

    def __str__(self) -> str:
        """
        Returns a summary of the pipline
        Args:
            None
        Returns:
            a summary of the pipline[str]
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Returns the model in the pipline
        Args:
            None
        Returns:
            The model[Model] in the pipline
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during
        the pipeline execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(
            name="pipeline_config", data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        Registers the artifact
        (adds the artifact to dictionary, self._artifacts).
        Args:
            name[str]: The name of the artifact
            artifact[Artifact]: The artifact that needs to be added.
        Returns:
            None
        """
        self._artifacts[name] = artifact

    def save_as_artifact(self, name: str, version: str) -> None:
        """
        Saves the pipline as an artifact.
        Args:
            name[str]: The name of the pipline.
            version[str]: The version of the pipline.
        Returns:
            None
        """

        saved_pipeline = Artifact(
            name=name, type="pipeline",
            data=self._dataset.data,
            asset_path=Path(f"pipelines/{name}"),
            metadata={
                "metrics": [metric.name for metric in self._metrics],
                "model": self.model.name,
                "input_features": (
                    [feature.name for feature in self._input_features]),
                "target_feature": self._target_feature.name,
                "split": self._split},
            version=version)
        saved_pipeline.save(saved_pipeline.data)

        return saved_pipeline

    @staticmethod
    def load_from_artifact(artifact: Artifact) -> "Pipeline":
        """
        Loads the pipline from artifact.
        Args:
            artifact[Artifact]: The artifact
            that should be loaded into a pipline.
        Returns:
            Pipline[Pipline]: The pipline loaded from artifact.
        """

        saved_data = artifact.metadata
        list_of_features = detect_feature_types_from_dataframe(
            pd.read_csv(io.StringIO(artifact.data.decode()))
        )
        input_features = [feature for feature in list_of_features
                          if feature.name in saved_data["input_features"]]
        target_feature = [feature for feature in list_of_features
                          if feature.name == saved_data["target_feature"]][0]

        return Pipeline(
            [get_metric(metric)
                for metric in saved_data["metrics"]],
            artifact.data,
            get_model(saved_data["model"]),
            input_features,
            target_feature,
            saved_data["split"])

    def _preprocess_features(self) -> None:
        """
        Preprocesses the features given at creation.
        Args:
            None
        Returns:
            None
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        if self.model.type == "classification":
            do_not_be = "k_nearest_neighbours"
            if target_data.ndim > 1 and self.model.name != do_not_be:
                target_data = target_data.argmax(1)
                self._collapsed_y = True
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        """
        Split the data given at creation into training and testing sets.
        Args:
            None
        Returns:
            None
        """
        split = self._split
        self._train_X = (
            [vector[:int(
                split * len(vector))] for vector in self._input_vectors]
        )
        self._test_X = [vector[
            int(split * len(vector)):] for vector in self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Joins multiple vectors together.
        Args:
            vectors[vectors: List[np.array]]
        Returns:
            A vector[np.array]
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model given at creation(in self).
        After that it also predicts using the testing sets.
        Args:
            None
        Returns:
            None
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._training_metrics_results.append((metric, result))

    def _evaluate(self) -> None:
        """
        Evaluates all metrics given at creation (in self._metrics)
        Args:
            None
        Returns
            None
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._evaluation_metrics_results.append((metric, result))
        if self._collapsed_y:
            pass
            # original_data = self._dataset.read()[self._target_feature.name]
            # predictions = original_data[predictions]
            # print("LOOK HERE!!!!!!!!!!!!!!!!!!")
        self._predictions = predictions

    def execute(self) -> None:
        """
        Executes the pipeline: preprocesses and splits the data,
        trains the model and then evaluates using the metrics.
        Args:
            None
        Return:
            None
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "training_metrics": self._training_metrics_results,
            "evaluation_metrics": self._evaluation_metrics_results,
            "predictions": self._predictions,
        }
