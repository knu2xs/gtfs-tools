from datetime import datetime
from importlib.util import find_spec
import json
import re
import unicodedata

__all__ = ["has_arcpy", "slugify"]

# provide variable indicating if arcpy is available
has_arcpy: bool = find_spec("arcpy") is not None


def slugify(value, replacement_character: str = "_", allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to ``replacement_character``; default is ``_``. Remove characters
    that aren't alphanumerics, underscores, or hyphens. Convert to lowercase.
    Also strip leading and trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    value = re.sub(r"[-\s]+", replacement_character, value).strip("-_")
    return value
