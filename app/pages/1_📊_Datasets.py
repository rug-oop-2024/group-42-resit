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

    def create(self, full_asset_path: Path) -> Dataset:
        """
        creates a dataset and saves it given its asset path
        Args:
            full_asset_path[str]: the full asset path to the csv file
        Return:
            A saved dataset
        """

        name = os.path.split(full_asset_path)[1]

        df = pd.read_csv(full_asset_path)

        # st.write()

        dataset = Dataset.from_dataframe(
            name=os.path.basename(name),
            asset_path=os.path.relpath(
                full_asset_path, "assets/objects/datasets"),
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
        automl.registry.delete(artifact.id)
        os.remove("./assets/dbo/artifacts/" + artifact.id)
        st.write("file deleted")
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

dataset = None
path = st.selectbox("Select a dataset", options, None)
uploaded_file = st.file_uploader("Or upload a .csv file", type=["csv"])

if uploaded_file is not None and uploaded_file.type == "text/csv":
    df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode()))
    dataset = Dataset.from_dataframe(
        df, uploaded_file.name, uploaded_file.name)
    dataset.save(dataset._data)
    management.save(dataset)

if st.button("delete file") and path is not None:
    normpath = os.path.normpath(path)
    management.delete(management.create(normpath))

if path is not None:
    normpath = os.path.normpath(path)
    dataset = management.create(normpath)

if dataset is not None:
    st.dataframe(dataset.read())
