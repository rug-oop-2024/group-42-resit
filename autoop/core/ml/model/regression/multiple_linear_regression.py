""".

File: multiple_linear_regression.py
Authors: Oscar Schuyl(s5576954, o.schuyl@student.rug.nl)
         & Santos Bril (s5529875, s.l.bril@student.rug.nl)
Description: a file that implements;
a class for a multiple linear regression model.

Adapted from exercise 1 of the oop course
"""

from copy import deepcopy

import numpy as np
from pydantic import PrivateAttr

from autoop.core.ml.model.model import RegressionModel


class MultipleLinearRegression(RegressionModel):
    """A model for multiple linear regression."""
    _slope: np.ndarray = PrivateAttr(default=np.ndarray)

    @property
    def get_slope(self) -> np.ndarray:
        """
        Getter for slope
        Args:
            None
        Returns:
            The slope np.ndarray.
        """
        return deepcopy(self._slope)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Uses the observations and ground truth
        to create a slope for prediction.
        Args:
            observations[np.ndarray]: The observations of the training data.
            ground_truth[np.ndarray]: The ground truth of the training data.
        Returns:
            None
        """
        super().fit(observations, ground_truth)
        self._observations = observations
        self._ground_truth = ground_truth
        observations_with_constants = (
            MultipleLinearRegression._add_row_of_ones(
                observations
            )
        )
        # reduce amount of math necessary as this happens twice
        transposed_observations = observations_with_constants.T

        first_part_equation = np.linalg.inv(
            transposed_observations @ observations_with_constants
        )
        second_part_equation = transposed_observations @ ground_truth
        self._slope = first_part_equation @ second_part_equation
        self._parameters.update({"slope": self._slope})

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Using the slope and observations to calculate
        and return the ground truth.
        Args:
            observations[np.ndarray]:
                The observations that need to be predicted
        Returns:
            The predictions of the model as an np.ndarray.
        """
        super().predict(observations)
        if type(self._slope) is not np.ndarray:
            variable_type = "The variable self._slope is type"
            variable_type_2 = f"{type(self._slope)}"
            variable_type_3 = ", self._slope should be np.ndarray."
            raise TypeError(variable_type + variable_type_2 + variable_type_3)
        if observations.shape[1] is not len(self._slope) - 1:
            columns = "Unexpected amount of columns in observations,"
            expected = f"expected{len(self._slope) - 1}"
            got = f" got{observations.shape[1]}."
            raise ValueError(columns + expected + got)

        observations_with_constants = (
            MultipleLinearRegression._add_row_of_ones(
                observations
            )
        )
        return observations_with_constants @ self._slope

    def _add_row_of_ones(matrix: np.ndarray) -> np.ndarray:
        """
        Adds a row of ones to the matrix.
        Args:
            matrix[np.ndarray]: This matrix will get an additional row of ones.
        Returns:
            Matrix[np.ndarray] with an added row of ones.
        """
        if type(matrix) is not np.ndarray:
            variable_type = f"The variable matrix is type {type(matrix)},"
            raise TypeError(variable_type + "matrix should be np.ndarray.")

        matrix_with_rows_of_one = np.append(
            matrix, np.array([[1] for _ in range(matrix.shape[0])]), axis=1
        )
        return matrix_with_rows_of_one
