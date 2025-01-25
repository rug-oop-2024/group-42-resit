import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import (
    CATEGORICAL_METRICS, CONTINUOUS_METRICS, get_metric)
from autoop.core.ml.model import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS, get_model)
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types
from copy import deepcopy
import os

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    writes helper text
    Args:
        text[str]: input text
    Returns:
        None
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")

helper_text_1 = "In this section, you can design a machine learning pipeline"
write_helper_text(helper_text_1 + " to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

names = [item.name for item in datasets]
chosen_model = None
current_metrics = None
current_model = None
chosen_target = None
chosen_features = None
pipeline = None

current_dataset_name = st.selectbox("Select a dataset", names, None)

if current_dataset_name:
    current_dataset = Dataset.from_artifact(
        datasets[names.index(current_dataset_name)])
    list_of_features = detect_feature_types(current_dataset)

if current_dataset_name is not None and chosen_target is None:
    chosen_target = st.selectbox(
        "choose target feature", list_of_features, None)

if not (chosen_target is None or chosen_target == []):
    features_of_target_type = [
        feature for feature in list_of_features if feature.type == chosen_target.type
        and feature.name != chosen_target.name]
    chosen_features: list[Feature] = st.multiselect(
        "choose input",
        features_of_target_type)

# if chosen_target != None and chosen_target.type == "categorical":
#     raw = deepcopy(current_dataset.read())
#     data = deepcopy(raw[chosen_target.name])
#     types = []
#     for row in data:
#         if row not in types and type(row) is str:
#             types.append(row)
#     categorical_done = True
#     chosen_type = st.selectbox("select the chosen ... that you want", types, None)

if chosen_features:
    if chosen_target.type == "categorical":
        chosen_model = st.selectbox(
            "Select a model", CLASSIFICATION_MODELS, None)
    elif chosen_target.type == "continuous":
        chosen_model = st.selectbox("Select a model", REGRESSION_MODELS, None)
    else:
        st.write("Seems you somehow broke the program, well played")

if chosen_model is not None:
    current_model = get_model(chosen_model)
    chosen_split = st.slider("choose split", 0.1, 0.9, 0.8, step=0.01)
    if current_model.type == "classification":
        chosen_metrics: list = st.multiselect(
            "choose metrics (optional)", CATEGORICAL_METRICS, None)
    elif current_model.type == "regression":
        chosen_metrics: list = st.multiselect(
            "choose metrics (optional)", CONTINUOUS_METRICS, None)

    current_metrics = [get_metric(metric) for metric in chosen_metrics]

if current_model is not None:
    # if chosen_target.type == "categorical" and (chosen_type is None or chosen_type == ""):
    #     st.write("Categorical error: Make sure to choose a type for the categorical data")
    # else:
    #     chosen_dataset = current_dataset
    #     if chosen_target.type == "categorical":
    #         for index in range(len(data)):
    #             if data[index] == chosen_type:
    #                 data[index] = 1
    #             else:
    #                 data[index] = 0
    #         raw[chosen_target.name] = data
    #         st.write(raw)
    #         chosen_dataset._data = raw.to_csv(index=False).encode()
        pipeline = Pipeline(current_metrics,
                            current_dataset,
                            current_model,
                            chosen_features,
                            chosen_target,
                            chosen_split)
        st.write(pipeline)
        st.write(pipeline.execute())

if pipeline:
    name = st.text_input("give name")
    version = st.text_input("give version")
    if st.button("save pipeline"):
       saved_pipeline = pipeline.save_as_artifact(name, version)
       automl.registry.register(saved_pipeline)
       st.write("saved succesfully")