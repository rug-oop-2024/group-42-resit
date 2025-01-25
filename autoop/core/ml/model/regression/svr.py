from sklearn.svm import SVR
from copy import deepcopy

import numpy as np
from pydantic import PrivateAttr

from autoop.core.ml.model.model import RegressionModel


class Support_Vector_Regression(RegressionModel):
    """
    A class that acts as a wrapper for the
    ensemble from scikit-learn.ensemble
    """

    _name: str = PrivateAttr("support_vector_regression")

    _instance_of_svr: SVR = (
        PrivateAttr(default=SVR())
    )

    @property
    def svc(self) -> SVR:
        """
        Getter for the instance of random forest classifier.
        Args:
            None
        Return:
            An instance of Random Forest classifier
                [ensemble.RandomForestClassifier]
        """
        return deepcopy(self._instance_of_svr)

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
        self._instance_of_svr.fit(
            observations, ground_truth)
        self._parameters = self._parameters | self._instance_of_svr.get_params()

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
        return self._instance_of_svr.predict(observations)
