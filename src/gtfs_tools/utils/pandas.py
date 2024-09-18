import numpy as np
import pandas as pd


def replace_zero_and_space_strings_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all zero length or all space strings in string columns with NaNs.

    Args:
        df: Pandas DataFrame.
    """
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df
