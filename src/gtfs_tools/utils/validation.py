import importlib.util
from typing import Union, Optional
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .gtfs import get_route_types_table

__all__ = ["validate_modality_codes", "validate_required_files"]


def validate_required_files(
    gtfs: Union[str, Path, "GtfsDataset"],
    required_files: Optional[list[str]],
    calendar_or_calendar_dates: Optional[bool] = True,
) -> bool:
    """
    Ensure all necessary files are present. Required files are determined by the ``required_files`` construction
    parameter.

    Args:
        gtfs: The GTFS dataset to check for required files.
        required_files: The list of required file names to validate.
        calendar_or_calendar_dates: When scanning for files, accept the dataset if ``calendar_dates.txt`` is
            present, even in the absence of ``calendar.txt``.
    """
    # late import to avoid circular imports
    from gtfs_tools.gtfs import GtfsDataset

    # get the directory to work with
    if isinstance(gtfs, str):
        gtfs_pth = Path(gtfs)
    elif isinstance(gtfs, GtfsDataset):
        gtfs_pth = gtfs.gtfs_folder
    elif isinstance(gtfs, Path):
        gtfs_pth = gtfs
    else:
        raise ValueError("gtfs must be either a path or a GtfsDataset")

    # ensure can even locate the directory to find the files
    if not gtfs_pth.exists():
        logging.error(
            f"Cannot locate provided path to GTFS dataset directory {gtfs_pth}"
        )

    # collect required files based on context
    if isinstance(gtfs, GtfsDataset) and required_files is None:
        required_files = gtfs.required_files
    else:
        required_files = [
            "agency.txt",
            "routes.txt",
            "shapes.txt",
            "stops.txt",
            "stop_times.txt",
            "trips.txt",
        ]

    # pull out the required file names without the extension
    req_lst = [f.rstrip(".txt") for f in required_files]

    # get files present in the dataset directory
    file_lst = [f.stem for f in gtfs_pth.glob("**/*.txt")]

    # determine if multiple gtfs datasets are in the child directory based on the required agency table
    agency_lst = [f for f in file_lst if f == "agency"]
    if len(agency_lst) > 1:
        logging.error(
            f"More than one GTFS dataset (agency file) detected in {gtfs_pth}. Can only load one GTFS "
            f"dataset at a time."
        )
        valid = False

    # if handling calendar and calendar dates separately
    if calendar_or_calendar_dates:
        # list of the potential valid calendar files
        cal_lst = ["calendar", "calendar_dates"]

        # check for the presence of the calendar or calendar_dates file
        cal_res = [fl for fl in file_lst if fl in cal_lst]

        # if either found, update required files to no longer include the calendar files
        if len(cal_res) > 0:
            # update the required files to no longer include calendar
            req_lst = [f for f in file_lst if f not in cal_lst]
            logging.debug(f"Required calendar files detected {cal_res}")

        else:
            valid = False
            logging.error("No required calendar files detected.")

    # check for the presence of all required files
    missing_lst = [fl for fl in req_lst if fl not in file_lst]

    # if there are missing files, then invalid
    if len(missing_lst) == len(req_lst):
        valid = False
        logging.error(
            f"Cannot locate all required files in the GTFS dataset {gtfs_pth.stem}. Missing {missing_lst}."
        )

    elif len(missing_lst) > 0:
        valid = False
        logging.error(
            f"GTFS dataset, {gtfs_pth.stem}, is missing required files: {missing_lst}"
        )

    else:
        valid = True
        logging.debug(f"GTFS dataset, {gtfs_pth.stem}, includes all required files.")

    return valid


def _get_route_types(enforce_gtfs_strict: bool) -> pd.Series:
    """Helper function to get route types to evaluate for validation."""
    # collect values to use for validation
    typ_df = get_route_types_table()

    if enforce_gtfs_strict:
        typ_vals = typ_df["route_type_gtfs"]
    else:
        typ_vals = typ_df["route_type"]

    return typ_vals


def validate_modality_codes(
    routes_source: Union[pd.DataFrame, "GtfsDataset"],
    modality_codes_column: Optional[str] = "route_type",
    enforce_gtfs_strict: Optional[bool] = False,
) -> bool:
    """
    Ensure modality codes are valid values...not something strange.

    Args:
        routes_source: Source for routes. This can either be a data frame created from the ``routes.txt`` file, or a
            GTFS dataset object instance.
        modality_codes_column: Column in the GTFS dataset routes to check modality codes.
        enforce_gtfs_strict: Whether to use the strict interpretation of modality codes defined in the GTFS
            specification, or whether to allow the expanded European codes. The default is ``False``, to allow the
            European codes as well.
    """
    # late import to avoid circular imports
    from gtfs_tools.gtfs import GtfsDataset

    # determine routes source type and get necessary data
    if isinstance(routes_source, GtfsDataset):
        routes_df = routes_source.routes.data
    elif isinstance(routes_source, pd.DataFrame):
        routes_df = routes_source
    else:
        raise ValueError("routes_source must be either a data frame or a GtfsDataset")

    # get a  list of route type values to compare against
    typ_vals = _get_route_types(enforce_gtfs_strict)

    # evaluate all values
    valid = routes_df[modality_codes_column].isin(typ_vals).all()

    if valid:
        logging.debug("All modality codes (route types) are valid.")
    else:
        logging.warning("Not all modality codes (route types) are valid.")

    return valid


def _validate_stop(row: pd.Series, enforce_gtfs_strict: bool) -> tuple[bool, str]:
    """
    Validate a single record and return the validation status along with a string indicating what the issue is.

    Args:
        row: GTFS stop row to validate.
        enforce_gtfs_strict: Whether to use the strict interpretation of stop codes.
    """

    # list to store error messages
    msg_lst = []

    # ensure coordinates are not null and not out of bounds
    if row["stop_lon"] is None or np.isnan(row["stop_lat"]):
        msg_lst.append("stop_lon is Null")

    elif -180 > row["stop_lon"] > 180:
        msg_lst.append("stop_lon out of bounds (+-180)")

    if row["stop_lat"] is None or np.isnan(row["stop_lon"]):
        msg_lst.append("stop_lat is Null")

    elif -90 > row["stop_lat"] > 90:
        msg_lst.append("stop_lat out of bounds (+-90)")

    # handle if stops types (modality) not yet added to schema
    if "route_type" in row.index.to_list():
        # get a list of route types to compare against
        typ_vals = _get_route_types(enforce_gtfs_strict)

        # ensure a route type can be inferred from stops > stop_times > trips > routes
        if row["route_type"] == np.nan or row["route_type"] not in typ_vals:
            msg_lst.append("cannot determine route_type (modality)")

    # format the return tuple based on if any issues were discovered
    if len(msg_lst) > 0:
        valid = False
        msg = ", ".join(msg_lst)
    else:
        valid = True
        msg = None

    return valid, msg


def validate_stop_rows(
    stops_df: pd.DataFrame,
    enforce_gtfs_strict: Optional[bool] = False,
    copy: Optional[bool] = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate each row in the stops data frame.

    Args:
        stops_df: GTFS stops dataframe to validate.
        enforce_gtfs_strict: Whether to use the strict interpretation of stop codes.
        copy: Whether to copy the stops dataframe. If not, the data frame will be modified in-place.

    Returns:
        A tuple of two data frames. One with
    """
    # cache needed properties for re-enabling spatially enable data frame at end
    sedf = stops_df.spatial.validate()
    geom_col = stops_df.spatial.name

    # copy the data frame if desired
    if copy:
        stops_df = stops_df.copy(deep=True)

    # ensure needed columns are present
    needed_cols = ["stop_lon", "stop_lat"]
    missing_cols = [c for c in needed_cols if c not in stops_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            f"The stops_df is missing needed columns for row validation {missing_cols}"
        )

    # if there are stops to work with
    if stops_df.shape[0] > 0:
        # run validation on each row
        stops_df[["valid", "error_messages"]] = stops_df.apply(
            lambda row: _validate_stop(row, enforce_gtfs_strict),
            axis="columns",
            result_type="expand",
        )

        # prune the table to just the invalid records
        invalid_df = stops_df.loc[~stops_df["valid"]]

        # if there are invalid stops
        if len(invalid_df.index) > 0:
            # get counts
            invalid_cnt = len(invalid_df.index)

            # report issues
            logging.warning(f"{invalid_cnt:,} stops are invalid.")

        # copy any invalid records to a separate data frame
        invalid_df = stops_df.loc[~stops_df["valid"]]

        # prune the stops to just valid records
        stops_df = stops_df[stops_df["valid"]].drop(columns=["valid", "error_messages"])

        # if working with a spatially enabled data frame, enable it
        if sedf:
            stops_df.spatial.set_geometry(geom_col, inplace=True)

    # ensure there is an invalid table to return
    else:
        invalid_df = stops_df.copy(deep=True)
        invalid_df[["valid", "error_messages"]] = [None, None]

    return stops_df, invalid_df
