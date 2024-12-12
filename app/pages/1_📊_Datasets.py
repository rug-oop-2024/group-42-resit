import io
import os
from glob import glob
from pathlib import Path

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset


class Management():
    """
    Management class, handles management of artifacts.
    """

    def create(self, full_asset_path: str) -> Dataset:
        """
        creates a dataset and saves it given its asset path
        Args:
            full_asset_path[str]: the full asset path to the csv file
        Return:
            A saved dataset
        """
        data_path = Path(full_asset_path)
        df = pd.read_csv(data_path)

        name_and_asset_path = full_asset_path.split("\\")[-1:][0]
        dataset = Dataset.from_dataframe(
            name=name_and_asset_path,
            asset_path=name_and_asset_path,
            data=df,
        )
        self.save(dataset)
        return dataset

    def delete(self, artifact: Artifact) -> None:
        """
        Deletes an artifact from the database and storage.
        Args:
            artifact[Artifact]: The artifact that needs to be deleted.
        Returns:
            None
        """
        st.write("file deleted")
        automl.registry.delete(artifact.id)
        os.remove("./assets/dbo/artifacts/" + artifact.id)
        st.rerun()

    def save(self, artifact: Artifact) -> None:
        """
        Saves an artifact
        Args:
            artifact[Artifact]: The artifact that needs to be saved.
        Returns:
            None
        """
        instance_of_automl = AutoMLSystem.get_instance()
        instance_of_automl.registry.register(artifact)


automl = AutoMLSystem.get_instance()

management = Management()

options = glob("**/*.csv", recursive=True)

path = st.selectbox("Select a dataset", options)
uploaded_file = st.file_uploader("Or upload a .csv file")

if uploaded_file is not None and uploaded_file.type == "text/csv":
    df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode()))
    dataset = Dataset.from_dataframe(
        df, uploaded_file.name, uploaded_file.name)
    management.save(dataset)
if st.button("delete file") and path is not None:
    management.delete(management.create(path))

if path is not None:
    dataset = management.create(path)

datasets = automl.registry.list(type="dataset")
st.dataframe(dataset.read())
