from importlib.util import find_spec

__all__ = ["has_arcpy"]

# provide variable indicating if arcpy is available
has_arcpy: bool = find_spec("arcpy") is not None
