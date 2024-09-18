class MissingRequiredColumnError(KeyError):
    """
    Raised when a column is missing from source data when reading from a CSV file.
    """
