from typing import Optional

import numpy as np
import pandas as pd


def replace_zero_and_space_strings_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all zero length or all space strings in string columns with NaNs.

    Args:
        df: Pandas DataFrame.
    """
    # cache the columns with object (string) data types
    string_columns = [
        col
        for col in zip(df.columns, df.dtypes)
        if col[1].kind == "O" or col[1].kind == "string"
    ]

    # perform the string replacement
    df = df.replace(r"^\s*$", np.NaN, regex=True)

    # ensure the columns are the same data types when returned; if all null, can be cast to float64
    for dtype in string_columns:
        df[dtype[0]] = df[dtype[0]].astype(dtype[1])

    return df


def get_coefficient_of_variation(
    data: pd.DataFrame,
    groupby_column: str,
    column_to_calculate: str,
    keep_columns: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Calculate coefficient of variation for a column in a data frame.

    Args:
        data: Data frame with column to calculate coefficient for and column to group by.
        groupby_column: Column to group by.
        column_to_calculate: Column to calculate coefficient for.
        keep_columns: Whether to keep columns after calculating the coefficient of variation, mean and standard
            deviation.
    """
    # ensure columns are in the data
    if groupby_column not in data.columns:
        raise ValueError("The groupby column does not appear to be in the input data.")

    if column_to_calculate not in data.columns:
        raise ValueError(
            "The column_to_calculate column does not appear to be in the input data."
        )

    # create a groupby object to simplify calculation
    grp = data.groupby(groupby_column)

    # get the mean and standard devaition
    std_df = grp.std().rename(columns={column_to_calculate: "standard_deviation"})
    mean_df = grp.mean().rename(columns={column_to_calculate: "mean"})

    # combine the mean and standard deviation into a single data frame
    desc_df = mean_df.join(std_df)

    # calculate the coefficient of variation
    desc_df["coefficient_of_variation"] = (
        desc_df["standard_deviation"] / desc_df["mean"]
    )

    # if only returning coefficient of variation
    if not keep_columns:
        desc_df.drop(columns=["standard_deviation", "mean"], inplace=True)

    return desc_df


def timedelta_to_minutes(timedelta: pd.Timedelta) -> pd.Timedelta:
    """Calculate the number of decimal minutes in a timedelta object."""
    return timedelta.total_seconds() / 60


def encode_cyndrical_features(
    df: pd.DataFrame, col: str, max_val: Optional[float] = None
) -> pd.DataFrame:
    """Encode cyndrical features with sine and cosine."""
    # https://medium.com/@axelazara6/why-we-need-encoding-cyclical-features-79ecc3531232
    # https://www.thedataschool.com.au/ryan-edwards/feature-engineering-cyclical-variables/

    # get the maximum value for the column if not provided
    if max_val is None:
        max_val = df[col].max()

    # calculate the sine and cosine of the column
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)

    return df
