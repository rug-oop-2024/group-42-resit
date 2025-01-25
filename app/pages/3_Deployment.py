import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

st.set_page_config(page_title="Deployment")

automl = AutoMLSystem.get_instance()

saved_pipelines = automl.registry.list("pipeline")

pipeline_names = [f"file: {pipeline.name}, version:{pipeline.version}"
                  for pipeline in saved_pipelines]

chosen_name = st.selectbox("Select a pipeline", pipeline_names)

chosen_pipeline = saved_pipelines[pipeline_names.index(chosen_name)]

pipeline = Pipeline.load_from_artifact(chosen_pipeline)

st.write(pipeline)

datasets = automl.registry.list(type="dataset")

dataset_names = [item.name for item in datasets]
current_dataset_name = st.selectbox("Select a dataset", dataset_names, None)

if current_dataset_name:
    current_dataset = Dataset.from_artifact(
        datasets[dataset_names.index(current_dataset_name)])
    list_of_features = detect_feature_types(current_dataset)
