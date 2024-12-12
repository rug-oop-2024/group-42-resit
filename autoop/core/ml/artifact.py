import base64
import os
from copy import deepcopy


class Artifact():
    """
    Artifact class that allows for saving and reading data from files
    Methods: save, remove, read
    """

    def __init__(self,
                 type: str, name: str, data: object,
                 asset_path: str, tags: list = [],
                 metadata: dict = {}, version: str = "1.0.0") -> None:
        """
        initialises init and creates an id based on the asset path
        requirements: type, name, data, asset_path
        optional add-ons: tags, metadata, version
        """

        self._name = name
        self._type = type
        self._asset_path = asset_path
        self._data = data
        self._tags = tags
        self._metadata = metadata
        self._version = version
        encoded = base64.b64encode(asset_path.encode("utf-8"))
        encoded_string = encoded.decode("utf-8")
        self._id = f"{encoded_string}={version}"

        self.save(self._data)

    @property
    def name(self) -> str:
        """
        returns the name of the Artifact
        """
        return self._name

    @property
    def type(self) -> str:
        """
        returns the type of the Artifact
        """
        return self._type

    @property
    def asset_path(self) -> str:
        """
        returns the asset path of the Artifact
        """
        return self._asset_path

    @property
    def data(self) -> object:
        """
        returns a deepcopy of the data of the Artifact
        """
        return deepcopy(self._data)

    @property
    def tags(self) -> list:
        """
        returns a deepcopy of the tags of the Artifact
        """
        return deepcopy(self._tags)

    @property
    def metadata(self) -> dict:
        """
        returns a deepcopy of the metadata of the Artifact
        """
        return deepcopy(self._metadata)

    @property
    def version(self) -> str:
        """
        returns the version of the Artifact
        """
        return self._version

    @property
    def id(self) -> str:
        """
        returns the id of the Artifact
        """
        return self._id

    def save(self, bytes: bytes) -> None:
        """
        Saves the dataset in the datasets folder.
        If datasets doesn't exist yet will create the folder.
        Args:
            bytes[bytes]: Bytes of the dataset that need to be saved.
        Returns:
            None
        """

        if not os.path.exists("./assets/objects/"):
            os.makedirs("./assets/objects/", exist_ok=True)

        data = bytes.decode().split("\r")

        with open("./assets/objects/" + self._asset_path, "w") as file:
            file.writelines(data)

    def remove(self) -> None:
        """
        Removes the artifact.
        Args:
            None
        Returns:
            None
        """

        if os.path.exists("./assets/objects/" + self.asset_path):
            os.remove("./assets/objects/" + self.asset_path)

    def read(self) -> str:
        """
        Reads the data from the asset_path directory
        Args:
            None
        Returns:
            str
        """

        if not os.path.exists("./assets/objects/"):
            raise FileNotFoundError("assets directory not found")
        if not os.path.exists("./assets/objects/" + self._asset_path):
            raise FileNotFoundError(f"{self._asset_path} file not found")
        try:
            with open("./assets/objects/" + self._asset_path, "r") as file:
                return file.read()
        except ValueError:
            error = "couldn't import from save, the file might be corrupted"
            raise ValueError(error)
