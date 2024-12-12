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

current_dataset_name = st.selectbox("Select a dataset", names)

if current_dataset_name:
    current_dataset = Dataset.from_artifact(
        datasets[names.index(current_dataset_name)])
    list_of_features = detect_feature_types(current_dataset)

    chosen_features: list[Feature] = st.multiselect(
        "choose features", list_of_features)

    chosen_preffered_feature = st.selectbox(
        "choose preffered feature", chosen_features, None)

if chosen_preffered_feature:
    if chosen_preffered_feature.type == "categorical":
        chosen_model = st.selectbox(
            "Select a model", CLASSIFICATION_MODELS, None)
    elif chosen_preffered_feature.type == "continuous":
        chosen_model = st.selectbox("Select a model", REGRESSION_MODELS, None)
    else:
        st.write("I... don't know... how you got here?")

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

if current_model is not None and st.button("run"):
    pipeline = Pipeline(current_metrics,
                        current_dataset,
                        current_model,
                        list_of_features,
                        chosen_preffered_feature,
                        chosen_split)
    st.write(pipeline)
    st.write(f"pipeline split: {pipeline._split}")

    # pipeline._preprocess_features()
    # pipeline._split_data()
    # pipeline._train()
    st.write(pipeline.execute())
