"""
init for the classification models
"""
from .k_nearest_neighbors import KNearestNeighbors
from .logistic_regression_wrapper import LogisticRegressionModel
from .random_forest_classifier_wrapper import RandomForestClassifier

models = [KNearestNeighbors, LogisticRegressionModel, RandomForestClassifier]
