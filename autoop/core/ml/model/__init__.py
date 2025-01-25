"""
Init for the classification and regression models.
"""
from .classification import (
    KNearestNeighbors,
    LogisticRegressionModel,
    RandomForestClassifier,
    Support_Vector_Machine
)
from .model import Model
from .regression import ( 
    Lasso, 
    MultipleLinearRegression,
    RandomForestRegressor,
    Support_Vector_Regression
)
REGRESSION_MODELS = [
    "multiple_linear_regression",
    # "random_forest_regressor",
    "lasso",
    "support_vector_regression"
]

CLASSIFICATION_MODELS = [
    "logistic_regression",
    # "random_forest_classifier",
    "k_nearest_neighbours" ,
    "support_vector_machine"
]


def get_model(model_name: str) -> Model:
    """
    Factory function to get a model by name.
    Args:
        model_name[str]: The name of a model that needs to be retrieved.
    Returns:
        The model[Model] of the given name.
    """
    lowercase_model_name = model_name.lower()
    error_message_1 = f"We didn't implement {lowercase_model_name}"
    error_message_2 = " in get_model yet, sorry"
    if lowercase_model_name in REGRESSION_MODELS:
        match lowercase_model_name:
            case "multiple_linear_regression":
                return MultipleLinearRegression()
            case "random_forest_regressor":
                print("RandomForestRegressor is unavailable"
                      +" due to a conflict between BaseModel and RandomForestRegressor")
                # return RandomForestRegressor()
            case "lasso":
                return Lasso()
            case "support_vector_regression":
                return Support_Vector_Regression()
            case _:
                raise NotImplementedError(error_message_1 + error_message_2)
    elif lowercase_model_name in CLASSIFICATION_MODELS:
        match lowercase_model_name:
            case "k_nearest_neighbours":
                return KNearestNeighbors()
            case "logistic_regression":
                return LogisticRegressionModel()
            case "random_forest_classifier":
                print("RandomForestClassifier is unavailable"
                      +" due to a conflict between BaseModel and RandomForestClassifier")
                # return RandomForestClassifier()
            case "support_vector_machine":
                return  Support_Vector_Machine()
            case _:
                raise NotImplementedError(error_message_1 + error_message_2)
    else:
        raise ValueError(f"no model by the name {lowercase_model_name}")
