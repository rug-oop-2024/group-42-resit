"""
Adapted from exercise 1 from the oop Course.
"""
from copy import deepcopy

import numpy as np
import sklearn.linear_model as linear
from pydantic import PrivateAttr

from autoop.core.ml.model.model import RegressionModel


class Lasso(RegressionModel):
    """
    A class that acts as a wrapper for the
    Lasso function from scikit-learn.linear_model.Lasso
    """

    _instance_of_lasso: linear.Lasso = PrivateAttr(default=linear.Lasso())

    @property
    def lasso(self) -> "Lasso":
        """
        Getter for lasso
        Args:
            None
        Returns:
            deepcopy of an instance of lasso
        """
        return deepcopy(self._instance_of_lasso)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Uses the observations and ground truth to create the intercept and
        coefficient for prediction. saves the values in self._parameters
        Args:
            observations[np.ndarray]: The observations of the training data.
            ground_truth[np.ndarray]: The ground truth of the training data.
        Returns:
            None
        """
        super().fit(observations, ground_truth)
        self._instance_of_lasso.fit(observations, ground_truth)
        self._parameters.update(
            {
                "_coef": self._instance_of_lasso.coef_,
                "_intercept": self._instance_of_lasso.intercept_,
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
        return self._instance_of_lasso.predict(observations)
