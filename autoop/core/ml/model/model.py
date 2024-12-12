
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from pydantic import BaseModel, PrivateAttr


class Model(ABC, BaseModel):
    """
    Base class for all metrics.
    """

    _type: str = PrivateAttr(default="Undefined")

    _parameters: dict = PrivateAttr(default=dict())

    @property
    def parameters(self) -> dict:
        """
        Getter for the parameters
        Args:
            None
        Returns:
            parameters[Dict{np.ndarray}
        """
        return deepcopy(self._parameters)

    @property
    def observations(self) -> np.ndarray:
        """
        Getter for the observations
        Args:
            None
        Returns:
            observations[np.ndarray]
        """
        return deepcopy(self._parameters["observations"])

    @property
    def ground_truth(self) -> np.ndarray:
        """
        Getter for the ground thruth
        Args:
            None
        Returns:
            ground truth[np.ndarray]
        """
        return deepcopy(self._parameters["ground_truth"])

    @property
    def type(self) -> str:
        """
        Getter for type
        Args:
            None
        Returns:
            type[string]
        """
        return self._type

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        A abstract method that fits(trains) the model on training data
        This abstract method does some common checks.
        Args:
            observations[np.ndarray]: The observations of the training data.
            ground_truth[np.ndarray]: The ground truth of the training data.
        Returns:
            None
        """
        error_1 = "fit doesn't accept observations of type:"
        if type(observations) is not np.ndarray:
            raise TypeError(error_1 + f"{type(observations)}")
        elif type(ground_truth) is not np.ndarray:
            raise TypeError(error_1 + f"{type(ground_truth)}")
        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Makes a prediction based on given observations.
        Args:
            observations[np.ndarray]:
                The observations that need to be predicted
        Returns:
            The predictions of the model as an np.ndarray.
        """
        if type(observations) is not np.ndarray:
            error_1 = "predict doesn't accept observations of type:"
            error_2 = f"{type(observations)}"
            raise TypeError(error_1 + error_2)


class ClassificationModel(Model):
    """
    Class for classifacation models,
    this makes it easier to distiguis between the two models.
    """

    _type: str = PrivateAttr("classification")


class RegressionModel(Model):
    """
    Class for Regression models,
    this makes it easier to distiguis between the two models.
    """

    _type: str = PrivateAttr("regression")
