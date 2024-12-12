from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from pydantic import BaseModel, PrivateAttr

from typechecker import Raise_Type_Error, Type_Checker

CONTINUOUS_METRICS = [
    "mean_squared_error",
    "mean_absolute_percentage_error",
    "root_mean_squared_error"
]

CATEGORICAL_METRICS = [
    "accuracy",
    "precision",
    "recall"
]


def get_metric(name: str) -> "Metric":
    """
    Gives a metric given its name.
    Args:
        name[str]: The name (in lowercase), of the metric that you want to get.
    Returns:
        The metric[Metric] with the given name.
    """

    lower_case_name = name.lower()
    error_message_1 = f"We didn't implement {lower_case_name}"
    error_message_2 = " in get_model yet, sorry"
    if lower_case_name in CONTINUOUS_METRICS:
        match lower_case_name:
            case "mean_squared_error":
                return MeanSquaredError()
            case "mean_absolute_percentage_error":
                return MeanAbsolutePercentageError()
            case "root_mean_squared_error":
                return RootMeanSquaredError()
            case _:
                raise NotImplementedError(error_message_1 + error_message_2)
    elif lower_case_name in CATEGORICAL_METRICS:
        match lower_case_name:
            case "accuracy":
                return Accuracy()
            case "precision":
                return Precision()
            case "recall":
                return Recall()
            case _:
                raise NotImplementedError(error_message_1 + error_message_2)
    else:
        raise ValueError(f"{lower_case_name} not in METRICS.")


class Metric(ABC, BaseModel):
    """
    Base class for all metrics.
    """

    _name: str = PrivateAttr(default="metric_base_class")

    _result: float = PrivateAttr(default=float)

    _data: dict = PrivateAttr(default=dict())

    _type: str = PrivateAttr(default="Undefined")

    @property
    def type(self) -> str:
        """
        Getter for the type of this class.
        Args:
            None
        Returns:
            type[str] (continuous, categorical or undefined)
        """
        return deepcopy(self._type)

    def __str__(self) -> str:
        """
        Summary of the results of the metric
        Args:
            None
        Returns:
            A summary[str]
        """
        return_str_1 = f"self.compared_items: {self._data}, result"
        return_str_2 = (return_str_1 + f" of {self._name}: {self._result}")
        return return_str_2

    def __call__(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Runs the evaluate method of this class(evaluates the metric)
        Args:
            prediction[np.ndarray]: The predicition of
                the model the metric is performed on.
            ground_truth[np.ndarray]:The actual data points.
        Returns:
            The outcome of the metric[float].
        """
        return self.evaluate(prediction, ground_truth)

    @abstractmethod
    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Abstract method of evaluate, checks for common errors
        and saves the data to self._data.
        Args:
            prediction (np.ndarray): prediction of the model
            ground_truth (np.ndarray): the actual data points
        Returns:
            None, children return a float
        """
        if not Type_Checker(prediction, np.ndarray):
            Raise_Type_Error(prediction, np.ndarray, "prediction")

        if not Type_Checker(ground_truth, np.ndarray):
            Raise_Type_Error(ground_truth, np.ndarray, "ground_truth")
        self._data = {"prediction": prediction,
                      "ground_truth": ground_truth}


class CategoricalMetric(Metric):
    """
    Class for categorical metrics, making it easiear for identification.
    """

    _type: str = PrivateAttr(default="categorical")


class ContinuousMetric(Metric):
    """
    Class for regression metrics, making it easiear for identification.
    """
    _type: str = PrivateAttr(default="continuous")


class Accuracy(CategoricalMetric):
    """
    Class for the accuracy metric.
    """
    _name: str = PrivateAttr("accuracy")

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Evaluates the Accuracy metric.
        Args:
            prediction[np.ndarray]: The predicition of
                the model the metric is performed on.
            ground_truth[np.ndarray]:The actual data points.
        Returns:
            The outcome of the metric[float].
        """
        super().evaluate(prediction, ground_truth)
        the_same_amount = sum(ground_truth == prediction)
        return the_same_amount / len(prediction)


class Precision(CategoricalMetric):
    """
    Class for hte precision metric
    """
    _name: str = PrivateAttr("precision")

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Evaluates the Precision metric.
        Args:
            prediction[np.ndarray]: The predicition of
                the model the metric is performed on.
            ground_truth[np.ndarray]:The actual data points.
        Returns:
            The outcome of the metric[float].
        """
        super().evaluate(prediction, ground_truth)

        true_positive_dict = {}
        false_negative_dict = {}

        for i in range(len(prediction)):
            prediction_string = prediction[i]
            ground_truth_string = ground_truth[i]
            if not Type_Checker(prediction_string, str):
                Raise_Type_Error(ground_truth_string, str, "prediction[i]")

            if not Type_Checker(prediction_string, str):
                Raise_Type_Error(ground_truth_string, str, "ground_truth[i]")

            if prediction_string not in true_positive_dict.keys():
                true_positive_dict.update({prediction_string: 0})
                false_negative_dict.update({prediction_string: 0})

            if ground_truth_string not in false_negative_dict.keys():
                true_positive_dict.update({ground_truth_string: 0})
                false_negative_dict.update({ground_truth_string: 0})

            if prediction_string == ground_truth_string:
                true_positive_dict[prediction_string] += 1
            else:
                false_negative_dict[prediction_string] += 1

        answer = 0
        for key in true_positive_dict.keys():
            answer += (true_positive_dict[key] / (
                true_positive_dict[key] + false_negative_dict[key]))

        return answer / len(true_positive_dict)


class Recall(CategoricalMetric):
    """
    Class for the recall metric
    """
    _name: str = PrivateAttr("recall")

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Evaluates the Recall metric.
        Args:
            prediction[np.ndarray]: The predicition of
                the model the metric is performed on.
            ground_truth[np.ndarray]:The actual data points.
        Returns:
            The outcome of the metric[float].
        """
        super().evaluate(prediction, ground_truth)

        true_positive_dict = {}
        false_positive_dict = {}

        for i in range(len(prediction)):
            prediction_string = prediction[i]
            ground_truth_string = ground_truth[i]
            if not Type_Checker(prediction_string, str):
                Raise_Type_Error(ground_truth_string, str, "prediction[i]")

            if not Type_Checker(prediction_string, str):
                Raise_Type_Error(ground_truth_string, str, "ground_truth[i]")

            if prediction_string not in true_positive_dict.keys():
                true_positive_dict.update({prediction_string: 0})
                false_positive_dict.update({prediction_string: 0})

            if ground_truth_string not in false_positive_dict.keys():
                true_positive_dict.update({ground_truth_string: 0})
                false_positive_dict.update({ground_truth_string: 0})

            if prediction_string == ground_truth_string:
                true_positive_dict[prediction_string] += 1
            else:
                false_positive_dict[ground_truth_string] += 1

        answer = 0
        for key in true_positive_dict.keys():
            answer += (true_positive_dict[key] / (
                true_positive_dict[key] + false_positive_dict[key]))

        return answer / len(true_positive_dict)


class MeanSquaredError(ContinuousMetric):
    """
    Class for mean squared error.
    """
    _name: str = PrivateAttr("mean_squared_error")

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Evaluates the Mean squared error metric.
        Args:
            prediction[np.ndarray]: The predicition of
                the model the metric is performed on
            ground_truth[np.ndarray]:The actual data points
        Returns:
            the outcome of the metric[float]
        """
        super().evaluate(prediction, ground_truth)
        error = np.subtract(ground_truth, prediction)
        squared_error = np.square(error)
        total_error = np.sum(squared_error)

        return total_error / len(prediction)


class RootMeanSquaredError(MeanSquaredError):
    """
    Class for Root mean squared error.
    """

    _name: str = PrivateAttr("root_mean_squared_error")

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        evaluates the root mean squared error
        Args:
            prediction (np.ndarray):prediction of the model
            ground_truth (np.ndarray): the actual data points
        Returns:
            answer (float): Percentage of error, the lower the better
        """
        mean_squared_error = super().evaluate(prediction, ground_truth)
        return np.sqrt(mean_squared_error)


class MeanAbsolutePercentageError(ContinuousMetric):
    """
    Class for mean absolute percentage error
    """

    _name: str = PrivateAttr("mean_absolute_percentage_error")

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        evaluates the mean absolute percantage error
        Args:
            prediction (np.ndarray):prediction of the model
            ground_truth (np.ndarray): the actual data points
        Returns:
            answer (float): Percentage of error, the lower the better
        """
        super().evaluate(prediction, ground_truth)
        error = np.divide(np.subtract(ground_truth, prediction), ground_truth)
        abs_error = np.abs(error)
        answer = np.sum(abs_error) / len(ground_truth) * 100
        return answer
