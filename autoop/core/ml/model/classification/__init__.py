"""
init for the classification models
"""
from .k_nearest_neighbors import KNearestNeighbors
from .logistic_regression_wrapper import LogisticRegressionModel
from .random_forest_classifier_wrapper import RandomForestClassifier
from .svm import Support_Vector_Machine

models_1 = [KNearestNeighbors, LogisticRegressionModel]
models_2 = [RandomForestClassifier, Support_Vector_Machine]
