def Type_Checker(value_to_be_checked: object,
                 type_to_compare_to: object) -> bool:
    """
    Check whether the imputted value is of the same type
    as the desired type and returns a bool.
    """
    return isinstance(value_to_be_checked, type_to_compare_to)


def Raise_Type_Error(value: object, type: type, item: str) -> None:

    """
    Raises a TypeError to reduce copy/pasting of code
    """
    string_type = f"item should be of type {str}, instead item is of type"
    other_type = f"{item} should be of type {type}, instead {item} is of type"
    if not Type_Checker(item, str):
        raise TypeError(string_type + f" {type(item)}")
    raise TypeError(other_type + f"{type(value)}")
