from pathlib import Path
import os
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from sklearn.datasets import load_iris #, fetch_openml
import pandas as pd
import numpy as np

iris = load_iris()

df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )

test = Dataset.from_dataframe(df, "iris.csv", "assets/objects")

print(iris.feature_names)
# print(df)
# print(test.read())