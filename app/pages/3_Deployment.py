import streamlit as st

from app.core.system import AutoMLSystem

st.set_page_config(page_title="Deployment")

automl = AutoMLSystem.get_instance()

saved_pipelines = automl.registry.list("pipeline")

pipeline = st.selectbox("Select a pipeline", saved_pipelines)
