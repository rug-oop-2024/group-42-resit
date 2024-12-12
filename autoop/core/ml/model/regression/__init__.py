"""
init for the regression models
"""
from .multiple_linear_regression import MultipleLinearRegression
from .random_forest_regressor_wrapper import RandomForestRegressor
from .sklearn_wrap import Lasso

list_models = [MultipleLinearRegression, RandomForestRegressor, Lasso]
