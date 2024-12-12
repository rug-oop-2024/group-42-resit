from typing import List

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Checks a given dataset for different types of features
    (categorical or continuous)
    Assumption: only categorical and continuous features and no NaN values.
    Args:
        dataset[Dataset]: The dataset that needs to be checked
    Returns:
        List[Feature]: List of features with their types.
    """

    pandaframe = dataset.read()
    featurelist = []

    for label, content in pandaframe.items():
        datatype = content.dtype.name
        feature = Feature(label)
        if datatype == "object":
            feature.type = "categorical"
            featurelist.append(feature)
        else:
            feature.type = "continuous"
            featurelist.append(feature)

    return featurelist
