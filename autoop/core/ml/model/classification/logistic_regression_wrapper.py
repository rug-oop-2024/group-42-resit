from copy import deepcopy

import numpy as np
import sklearn.linear_model as sk
from pydantic import PrivateAttr

from autoop.core.ml.model.model import ClassificationModel


class LogisticRegressionModel(ClassificationModel):
    """
    A class that acts as a wrapper for the
    Logisticregression function from scikit-learn.linear_model
    """

    _instance_of_logistic_regression: sk.LogisticRegression = (
        PrivateAttr(default=sk.LogisticRegression())
    )

    @property
    def logistic_regression(self) -> sk.LogisticRegression:
        """
        Getter for the instance of logistic regression.
        Args:
            None
        Return:
            An instance of logistic regression
                [sk.LogisticRegression]
        """
        return deepcopy(self._instance_of_logistic_regression)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Uses the observations and ground truth to create the intercept and
        coefficient for prediction. saves the values in self._parameters.
        Args:
            observations[np.ndarray]: The observations of the training data.
            ground_truth[np.ndarray]: The ground truth of the training data.
        Returns:
            None
        """
        super().fit(observations, ground_truth)
        self._instance_of_logistic_regression.fit(observations, ground_truth)
        self._parameters.update(
            {
                "_coef": self._instance_of_logistic_regression.coef_,
                "_intercept": self._instance_of_logistic_regression.intercept_,
            }
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the ground truth based on the observations,
        the intercept and the coefficient.
        Args:
            observations[np.ndarray]:
                The observations that need to be predicted
        Returns:
            The predictions of the model as an np.ndarray.
        """
        super().predict(observations)
        return self._instance_of_logistic_regression.predict(observations)
