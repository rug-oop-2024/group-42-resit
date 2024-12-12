from typing import List

from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import LocalStorage, Storage


class ArtifactRegistry():
    """
    Class for the registry of artifacts,
    handles saving and loading of artifacts
    """
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """
        Initializes Artifactregistry
        Args:
            database[Database]: database used for the registry of artifacts
            storage[Storage]: storage used for the registry of artifacts
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Saves the artifact in storage and the metadata in the database.
        Args:
            artifact[Artifact]: the artifact that needs to be saved.
        Returns:
            None
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact._version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }

        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Gives all the artifacts of a specified type
        Args:
            type[str]: type that you want the artifacts of
        Returns:
            List of artifacts
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact from the database.
        Args:
            Artifact_id[str]: Id of the artifact that needs to be retrieved.
        Returns:
            An Artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Removes an artifact from the database and the storage.
        Args
            artifact_id[str]: Id of an Artifact that needs to be removed.
        Returns
            None
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    AutoMLSystem class has both localstorage and database.
    """

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initializes AutoMLSystem
        Args:
            storage[LocalStorage]: The Local storage to be used.
            database[Database]: The database to be used.
        Returns:
            None
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Gives an instance of the AutoMLSystem class.
        Args:
            None
        Returns
            Instance of the AutoMLSystem class
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Getter for registry:
        Args:
            None
        Returns:
            self._registry [ArtifactRegistry]
        """
        return self._registry
