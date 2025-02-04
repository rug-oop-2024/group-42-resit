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
    features_of_target_type = []
    for feature in list_of_features:
        check_1 = (feature.type == chosen_target.type)
        check_2 = (feature.name != chosen_target.name)
        if check_1 and check_2:
            features_of_target_type.append(feature)
    chosen_features: list[Feature] = st.multiselect(
        "choose input",
        features_of_target_type)

if chosen_features:
    if chosen_target.type == "categorical":
        chosen_model = st.selectbox(
            "Select a model", CLASSIFICATION_MODELS, None)
    elif chosen_target.type == "continuous":
        chosen_model = st.selectbox("Select a model", REGRESSION_MODELS, None)
    else:
        st.write("Seems you somehow broke the program, well played")
    error_1 = "RandomForestRegressor and RandomForestClassifier"
    error_2 = " are unavailable due to a conflict with BaseModel"
    st.write(error_1 + error_2)

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
