from copy import deepcopy


class Feature():
    """
    Feature class either categorical and continuous.
    """
    def __init__(self, name: str, type: str = "Not identified") -> None:
        """
        Initilizes Feature
        Args:
            name[str]: the name of the feature
            type[str]: the type of the feature
                (continuous or categorical)
        Returns:
            None
        """
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        """
        Getter for name:
        Args:
            None
        Returns:
            name[str]
        """
        return deepcopy(self._name)

    @property
    def type(self) -> str:
        """
        Getter for type:
        Args:
            None
        Returns:
            type[str]
        """
        return deepcopy(self._type)

    @type.setter
    def type(self, new_type: str) -> None:
        """
        Setter for type, it will either become categorical or continuous
        Args:
            new_type[str]: The new type that the feature will have.
        Returns:
            None
        """
        if new_type in ["categorical", "continuous"]:
            self._type = new_type
        else:
            first_part = "type has to be categorical or continuous,"
            print(first_part + f" was given {new_type} instead")

    def __str__(self) -> str:
        """
        Gives a summary of feature, its name and type
        Args:
            None
        Returns:
            a summary[str]
        """
        return f"name: {self.name}\ntype: {self.type}"
