"""
init for the regression models
"""
from .multiple_linear_regression import MultipleLinearRegression
from .random_forest_regressor_wrapper import RandomForestRegressor
from .sklearn_wrap import Lasso
from .svr import Support_Vector_Regression

list_models_1 = [MultipleLinearRegression, RandomForestRegressor]
list_models_2 = [Lasso, Support_Vector_Regression]
