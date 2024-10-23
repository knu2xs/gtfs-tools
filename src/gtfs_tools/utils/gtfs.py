import datetime
import logging
import math
from pathlib import Path
import shlex
from typing import Optional, Iterable

import numpy as np
import pandas as pd

__all__ = [
    "add_agency_name_column",
    "add_location_descriptions",
    "add_modality_descriptions",
    "add_standardized_modality_column",
    "calculate_headway",
    "get_calendar_from_calendar_dates",
    "get_gtfs_directories",
    "get_route_types_table",
    "interpolate_stop_times",
    "standardize_route_types",
]


def get_gtfs_directories(parent_dir: Path) -> list[Path]:
    """
    Get a list of GTFS directories below a parent directory.

    .. note::
        This is based on searching for the required ``agency.txt`` file. Consequently, if this is missing,
        the GTFS directory will not be detected.

    Args:
        parent_dir (Path): Path to the parent directory to search in.
    """
    # ensure parent directory is a path
    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)

    # use list comprehension to search for required file, agency
    pth_lst = [pth.parent for pth in parent_dir.glob("**/agency.txt")]

    return pth_lst


def interpolate_stop_times(stop_times_df: pd.DataFrame) -> pd.DataFrame:
    """Helper to preprocess stop times' arrival and departure columns by handling missing values."""

    # flag for providing interpolation messaging
    stop_times_cnt = len(stop_times_df.index)

    # ensure data is correctly sorted
    stop_times_df.sort_values(["trip_id", "stop_sequence"], inplace=True)

    # apply interpolation to both arrival and departure columns
    for col in ["arrival_time", "departure_time"]:
        if col in stop_times_df.columns:
            # get the count to start working with
            df_cnt = (
                stop_times_df[["trip_id", col]]
                .groupby("trip_id")
                .count()
                .rename(columns={col: "count"})
            )

            # get the non-null count by trip to ascertain if valid
            df_valid = (
                stop_times_df[stop_times_df[col].notnull()][["trip_id", col]]
                .groupby("trip_id")
                .count()
                .rename(columns={col: "valid_count"})
            )

            # add the invalid count onto the valid count data frame
            df_valid = df_cnt.join(df_valid, on="trip_id", how="left")
            df_valid["invalid_count"] = df_valid["count"] - df_valid["valid_count"]

            # create data frame flags for valid and invalid trips
            df_valid["trip_valid"] = df_valid["valid_count"] >= 2
            df_valid["trip_invalid"] = df_valid["valid_count"] < 2

            # get the valid, invalid and update trip counts
            stop_valid_cnt = len(df_valid[df_valid["trip_valid"]].index)
            stop_invalid_cnt = len(df_valid[df_valid["trip_invalid"]].index)
            stop_update_cnt = (
                df_valid[df_valid["trip_valid"]]
                .reset_index()[["trip_id", "invalid_count"]]
                .drop_duplicates()["invalid_count"]
                .sum()
            )

            # if there are stop times to update, do it
            if stop_update_cnt > 0:
                # impute stop_times' times
                stop_times_df[col] = stop_times_df[col].interpolate().dt.round("1s")

            # only get super strict with arrival times, since these are the only required times
            if col == "arrival_time":
                # if there's nothing to work with
                if stop_valid_cnt == 0:
                    raise ValueError(f"No records in stop_times are usable.")

                # if there are valid records, and records will be interpolated
                elif stop_update_cnt > 0:
                    # notify stop times are being imputed
                    logging.info(
                        f"Intermediate times have been interpolated in the stop_times table "
                        f"({stop_update_cnt:,} interpolated / {stop_times_cnt:,} total)."
                    )

                # if there's something to work with
                elif stop_invalid_cnt > 0:
                    logging.warning(
                        f"Some records in stop_times_df do not have two times to use, and are unusable "
                        f"({stop_invalid_cnt:,} updated / {stop_times_cnt:,} total)."
                    )

    return stop_times_df


def get_calendar_from_calendar_dates(calendar_dates: pd.DataFrame) -> pd.DataFrame:
    """
    Infer a calendar data frame from the calendar dates.

    Args:
        calendar_dates: Data frame created from ``calendar_dates.txt``.
    """
    # only use exception type 1, when they are open
    calendar_dates = calendar_dates[calendar_dates["exception_type"] == 1]

    # cast the date column to a datetime object
    calendar_dates["date"] = pd.to_datetime(
        calendar_dates["date"].astype(str), format="%Y%m%d"
    )

    # get the day of the week from the datetime object
    calendar_dates["dow"] = calendar_dates["date"].dt.dayofweek

    # get the count for each service id for the day of the week
    df_dow = calendar_dates.groupby(["service_id", "dow"]).size().reset_index()
    df_dow.columns = ["service_id", "dow", "count"]

    # calculate a simple integer boolean for the day of the week from the count
    df_dow["status"] = df_dow["count"].astype(bool).astype(int)

    # add the day of the week onto the data frame
    df_dow_key = pd.DataFrame(
        [
            (0, "monday"),
            (1, "tuesday"),
            (2, "wednesday"),
            (3, "thursday"),
            (4, "friday"),
            (5, "saturday"),
            (6, "sunday"),
        ],
        columns=["dow", "dow_desc"],
    )

    df_dow = df_dow.join(df_dow_key.set_index("dow"), on="dow", how="left")

    # utilize the pivot table to reformat the table to show days of the week open
    df_cal = df_dow.pivot_table(
        values="status", columns="dow_desc", index="service_id"
    ).fillna(0)

    # account for possibility of missing days (Sunday commonly missing in New England states)
    for day_col in df_dow_key["dow_desc"]:
        if day_col not in df_cal.columns:
            df_cal[day_col] = 0

    # convert to integer
    for c in df_cal.columns:
        df_cal[c] = df_cal[c].astype(int)

    # reorganize the column order
    df_cal = df_cal[
        ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    ]

    # pull service_id out of the index
    df_cal.reset_index(inplace=True)

    # remove the status column name left over from the pivot
    df_cal.columns.name = None

    # pluck out current year and make a start and end year five years before and after
    yr = datetime.datetime.now().year
    start_year = yr - 5
    end_year = yr + 5

    # set the start and end date columns to be compliant
    # TODO: Potentially introspectively retrieve these from calendar_dates if eventually a business need arises
    df_cal["start_date"] = f"{start_year}0101"
    df_cal["end_date"] = f"{end_year}1231"

    return df_cal


# variable to cache route types table
_route_types_df = None


def get_route_types_table() -> pd.DataFrame:
    """Get a route types dataframe with translations between european codes and standard GTFS route type codes."""
    if _route_types_df is None:
        # create a data frame of descriptions for the route types
        route_type_pth = (
            Path(__file__).parent.parent / "assets" / "gtfs_modality_translation.csv"
        )

        route_type_df = pd.read_csv(
            filepath_or_buffer=route_type_pth,
            names=["route_type", "route_type_desc", "route_type_gtfs"],
            dtype={"route_type": str, "route_type_desc": str, "route_type_gtfs": str},
        )

    else:
        route_type_df = _route_types_df

    return route_type_df


def standardize_route_types(
    input_dataframe: pd.DataFrame, route_type_column: Optional[str] = "route_type"
) -> pd.DataFrame:
    """
    Replace any non-standard route types with standard GTFS route type codes.

    Args:
        input_dataframe: Data frame with route types to be standardized.
        route_type_column: Colum with route_types to be standardized. Defaults to ``route_types```.
    """
    # ensure the route type column exists in the input data
    if route_type_column not in input_dataframe.columns:
        raise ValueError(
            rf"The input dataframe does not contain the route_type column, {route_type_column}."
        )

    # get the crosstabs table
    lookup_df = get_route_types_table()[["route_type"]].rename(
        {"route_type_gtfs": route_type_column}, axis=1
    )

    # rename the existing column
    df = input_dataframe.rename(columns={route_type_column: "type_replace"})

    # join the lookup and drop the old
    out_df = df.join(lookup_df, on="type_replace", how="left")

    return out_df


def calculate_headway(stop_times_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the headway in minutes for each stop time in the stop_times dataframe.

    Args:
        stop_times_df: Stop times data frame.
    """
    # sort data by stop and arrival time, yielding a sequential list of arrivals for each stop
    col_lst = ["stop_id", "arrival_time"]
    sorted_df = stop_times_df.sort_values(col_lst).loc[:, col_lst]

    # add an incremental counter by stop; enables identifying first stop
    sorted_df["trip_idx"] = sorted_df.groupby("stop_id").cumcount() + 1

    # get the headway by getting the difference from the previous row to the current row
    sorted_df["headway"] = sorted_df["arrival_time"].diff()

    # remove the first trip from each stop (no previous value to calculate headway from) and zero values (when
    # two trips arrive at the same time)
    headway_df = sorted_df.loc[
        (sorted_df["trip_idx"] != 1) & (sorted_df["headway"] > pd.Timedelta(0)),
        ["stop_id", "arrival_time", "headway"],
    ]

    # convert headway to decimal minutes
    headway_df["headway"] = headway_df["headway"].apply(
        lambda td: td.total_seconds() / 60
    )

    return headway_df


def get_description_from_id(
    lookup_dataframe: pd.DataFrame,
    id_string: str,
    id_column: str,
    description_column: str,
    description_separator: str = ", ",
) -> str:
    # handle potentially other data types being shoved in
    if not isinstance(id_string, str) and id_string is not None:
        if math.isnan(id_string):
            id_string = None

        elif not math.isnan(id_string):
            if isinstance(id_string, float):
                id_string = int(id_string)

            elif isinstance(id_string, int):
                id_string = str(id_string)

    # only go to the effort if there is something to work with
    if id_string is None:
        std_str = None

    # make sure a zero length string is simply none
    elif len(id_string.replace(" ", "")) == 0:
        std_str = None
    else:
        # get the lookup table
        lookup = lookup_dataframe.set_index(id_column)[description_column]

        # ensure agency id is a string
        if not isinstance(id_string, str) and id_string is not None:
            id_string = str(id_string)

        # get the individual agency ids from the comma separated values, and account for string literals in quotes
        if id_string is not None:
            id_lst = shlex.split(id_string)

            # create a set of the route type standard codes to avoid duplicates, then convert to list for sorting
            std_lst = list(set(lookup.get(id) for id in id_lst))

            # remove any null values from list
            std_lst = [id for id in std_lst if id is not None]

        # combine descriptions into comma separated string if anything found
        if std_lst is None or len(std_lst) == 0:
            std_str = None
        else:
            std_str = description_separator.join(std_lst)

    return std_str


def add_description_column_from_id_column(
    data: pd.DataFrame,
    lookup_dataframe: pd.DataFrame,
    lookup_id_column: str,
    lookup_description_column: str,
    id_column: str,
    description_column: str,
    description_separator: str = ", ",
) -> pd.DataFrame:
    # ensure id column exists
    if id_column not in data.columns:
        raise ValueError(f"The dataframe does not contain the {id_column} column.")

    # ascertain if spatially enable dataframe or not
    spatial = data.spatial.validate()
    geom_col = data.spatial.name

    # make a copy so original data is not altered
    df = data.copy(deep=True)

    # populate the description column
    df[description_column] = df[id_column].apply(
        lambda id_val: get_description_from_id(
            lookup_dataframe,
            id_string=id_val,
            id_column=lookup_id_column,
            description_column=lookup_description_column,
            description_separator=description_separator,
        )
    )

    # if data originally spatial, re-enable it
    if spatial:
        df.spatial.set_geometry(geom_col, inplace=True)

    return df


def add_agency_name_column(
    data: pd.DataFrame,
    agency_df: pd.DataFrame,
    agency_id_column: Optional[str] = "agency_id",
    agency_name_column: Optional[str] = "agency_name",
) -> pd.DataFrame:
    """
    Add a standardized modality column to data frame. Some datasets can contain modality codes utilizing a much more
    detailed European standard for transit types. If a much more succinct coding is needed following the GTFS
    standard, this function will add a column with standardized modality codes.

    Args:
        data: Pandas data frame with a column containing agency identifiers.
        agency_df: Pandas data frame with agency identifiers and associated agency names.
        agency_id_column: Column containing agency identifiers. Default is ``agency_id``.
        agency_name_column: Column to be added with agency names. Default is ``agency_name``.
    """
    df = add_description_column_from_id_column(
        data,
        lookup_dataframe=agency_df,
        lookup_id_column="agency_id",
        lookup_description_column="agency_name",
        id_column=agency_id_column,
        description_column=agency_name_column,
    )
    return df


def _select_single_modality(type_string: str) -> str:
    """
    Lookup the highest priority transit modality from a list of integer values in a string separated by a comma.

    ref: https://gtfs.org/schedule/reference/#routestxt (specifically, the ``route_type`` column description)
    """
    # default to populate
    carto_typ = None

    # if a carto type is provided
    if type_string is not None:
        # split the types string on the comma into a list
        typ_lst = [itm.strip() for itm in type_string.split(",")]

        # get the remapping for the data types to the standard gtfs modality coding
        typ_recode = route_type_df["route_type_carto"]

        # look up the carto type to account for nonstandard modality coding
        typ_lst = set(int(typ_recode.loc[typ]) for typ in typ_lst)

        # hierarchy from lowest to highest
        typ_hierarchy = [
            4,  # ferry
            12,  # monorail
            6,  # aerial lift
            7,  # funicular
            2,  # rail
            1,  # subway
            0,  # light rail
            5,  # cable tram
            11,  # trolleybus
            3,  # bus
            31,  # school bus
        ]

        # if a type assigned
        if len(typ_lst):
            # iterate the hierarchical list to find the first match
            for typ in typ_hierarchy:
                if typ in typ_lst:
                    carto_typ = str(typ)
                    break

    return carto_typ


def add_standardized_modality_column(
    data: pd.DataFrame,
    modality_column: Optional[str] = "route_type",
    standardized_modality_column: Optional[str] = "route_type_std",
    assign_single_modality: bool = False,
) -> pd.DataFrame:
    """
    Add a standardized modality column to data frame. Some datasets can contain modality codes utilizing a much more
    detailed European standard for transit types. If a much more succicinct coding is needed following the GTFS
    standard, this function will add a column with standardized modality codes.
    Args:
        data: Pandas data frame with a column containing modality codes.
        modality_column: Column containing modality codes. Default is ``route_type``.
        standardized_modality_column: Column to be added with standardized route codes. Default is ``route_type_std``.
    """
    df = add_description_column_from_id_column(
        data,
        lookup_dataframe=get_route_types_table(),
        lookup_id_column="route_type",
        lookup_description_column="route_type_gtfs",
        id_column=modality_column,
        description_column=standardized_modality_column,
        description_separator=",",
    )

    if assign_single_modality:
        df[standardized_modality_column] = df[modality_column].apply(
            _select_single_modality
        )

    return df


def add_modality_descriptions(
    data: pd.DataFrame,
    modality_codes_column: Optional[str] = "route_type",
    description_column: Optional[str] = "route_type_desc",
) -> pd.DataFrame:
    """
    Add a string column with modality descriptions looked up from modality types.

    Args:
        data: Pandas data frame with a column containing modality codes.
        modality_codes_column: Column containing modality codes. Default is ``route_type``.
        description_column: Column to be added with route type descriptions. Default is ``route_type_desc``.
    """
    df = add_description_column_from_id_column(
        data,
        lookup_dataframe=get_route_types_table(),
        lookup_id_column="route_type",
        lookup_description_column="route_type_desc",
        id_column=modality_codes_column,
        description_column=description_column,
        description_separator=",",
    )
    return df


def add_location_descriptions(
    data: pd.DataFrame,
    location_codes_column: Optional[str] = "location_type",
    location_description_column: Optional[str] = "esri_location_type_desc",
) -> pd.DataFrame:
    """
    Add a string column with location descriptions looked up from location types.

    Args:
        data: Pandas data frame with a column containing location codes.
        location_codes_column: Column with location codes. Default is ``location_type``.
        location_description_column: Column to be added with location type descriptions. Default is
            ``esri_location_type_desc``.
    """
    # create a data frame for looking up
    location_type_df = pd.DataFrame(
        data=[
            ["0", "stop"],
            ["1", "station"],
            ["2", "entrance or exit"],
            ["3", "generic node"],
            ["4", "boarding area"],
        ],
        columns=["location_type", "location_type_desc"],
    )

    df = add_description_column_from_id_column(
        data,
        lookup_dataframe=location_type_df,
        lookup_id_column="location_type",
        lookup_description_column="location_type_desc",
        id_column=location_codes_column,
        description_column=location_description_column,
        description_separator=", ",
    )

    return df
