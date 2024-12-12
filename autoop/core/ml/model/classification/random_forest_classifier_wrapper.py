from copy import deepcopy

import numpy as np
import sklearn.ensemble as ensemble
from pydantic import PrivateAttr

from autoop.core.ml.model.model import ClassificationModel


class RandomForestClassifier(ClassificationModel):
    """
    A class that acts as a wrapper for the
    ensemble from scikit-learn.ensemble
    """

    _instance_of_random_forest_classifier: ensemble.RandomForestClassifier = (
        PrivateAttr(default=ensemble.RandomForestClassifier)
    )

    @property
    def random_forest_classifier(self) -> ensemble.RandomForestClassifier:
        """
        Getter for the instance of random forest classifier.
        Args:
            None
        Return:
            An instance of Random Forest classifier
                [ensemble.RandomForestClassifier]
        """
        return deepcopy(self._instance_of_random_forest_classifier)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Uses the observations and ground truth to fit (train) the model.
        saves the values in self._parameters
        Args:
            observations[np.ndarray]: The observations of the training data.
            ground_truth[np.ndarray]: The ground truth of the training data.
        Returns:
            None
        """
        super().fit(observations, ground_truth)
        self._instance_of_random_forest_classifier.fit(
            observations, ground_truth)
        self._parameters.update(
            {
                "estimators_": (
                    self._instance_of_random_forest_classifier.estimators_),
                "classes_": (
                    self._instance_of_random_forest_classifier.classes_),
            }
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        predicts the ground truth based on the observations,
        the intercept and the coefficient
        Args:
            observations[np.ndarray]:
                The observations that need to be predicted
        Returns:
            The predictions of the model as an np.ndarray.
        """
        super().predict(observations)
        return self._instance_of_random_forest_classifier.predict(observations)
