from pathlib import Path
import os
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from sklearn.datasets import load_iris #, fetch_openml
import pandas as pd
import numpy as np
from app.core.system import AutoMLSystem
from autoop.core.ml.metric import get_metric
from autoop.core.ml.model import get_model
from autoop.functional.feature import detect_feature_types

test = Artifact("adult.csv", None, "dataset", Path("datasets/adult.csv"))

test = Dataset.from_artifact(test)
test._data = test.read()

all_features = detect_feature_types(test)
chosen_target = None
chosen_features = []

for item in all_features:
    if item.type == "categorical":
        if chosen_target is None:
            chosen_target = item
            continue
        chosen_features.append(item)


pipeline = Pipeline([get_metric("accuracy")],
                    test,
                    get_model("logistic_regression"),
                    chosen_features,
                    chosen_target)

print(pipeline)
print(pipeline.execute())

# automl = AutoMLSystem.get_instance()

# datasets = automl.registry.list(type="dataset")

# print(datasets)