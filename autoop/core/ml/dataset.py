import io

import pandas as pd

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """
    Dataset class which extends Artifact, handles converting
    from artifact and from dataframe and reading the data.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        initialisation of a Dataset file
        sends all of them to Artifact
        with an extra parameter being type="dataset"
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0") -> "Dataset":
        """
        Converts a pandas dataframe to a dataset.
        Args:
            data[pd.DataFrame]: The dataframe that needs to be converted.
            name[str]: The name of the new data.
            asset_path[str]: The asset_path of the new data.
            version[str]: The version of the new data (default is 1.0.0)
        Returns:
            A Dataset converted from a the given data
        """

        dataset_2 = Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
        return dataset_2

    @staticmethod
    def from_artifact(artifact: Artifact) -> "Dataset":
        """
        Converts an artifact to a dataset.
        Args:
            artifact[Artifact]: The artifact that needs to be converted.
        Returns:
            A Dataset converted from a the given artifact
        """
        return Dataset(
            name=artifact.name,
            asset_path=artifact.asset_path,
            data=artifact.data,
            tags=artifact.tags,
            metadata=artifact.metadata,
            version=artifact.version
        )

    def read(self) -> pd.DataFrame:
        """
        Gives back the data in the dataset.
        Args:
            None
        Returns:
            A pandas dataframe converted from a the given artifact
        """
        csv = super().read()
        return pd.read_csv(io.StringIO(csv))
