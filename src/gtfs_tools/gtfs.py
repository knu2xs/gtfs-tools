import datetime
import importlib.util
from functools import cached_property
import logging
from pathlib import Path
import tempfile
from typing import Optional, Union
import zipfile

from cachetools import cached
from dask import dataframe as dd
import numpy as np
from arcgis.features import GeoAccessor, GeoDaskSpatialAccessor
from arcgis.geometry import Point, Polyline
import pandas as pd

from .utils.dt import hh_mm_to_timedelta
from .utils.exceptions import MissingRequiredColumnError
from .utils.gtfs import (
    get_calendar_from_calendar_dates,
    interpolate_stop_times,
    get_gtfs_directories,
    standardize_route_types as std_rt_typs,
    calculate_headway,
)
from .utils.pandas import replace_zero_and_space_strings_with_nulls
from .utils.validation import (
    validate_required_files,
    validate_modality_codes,
    validate_stop_rows,
)

__all__ = [
    "GtfsAgency",
    "GtfsCalendar",
    "GtfsCalendarDates",
    "GtfsDataset",
    "GtfsFile",
    "GtfsFrequencies",
    "GtfsRoutes",
    "GtfsShapes",
    "GtfsStops",
    "GtfsStopTimes",
    "GtfsTrips",
]


class GtfsFile(object):
    """
    Template object for child files in GTFS datasets.
    """

    # variables to be set in children
    required_columns: list[str] = []
    string_columns: list[str] = []
    integer_columns: list[str] = []
    float_columns: list[str] = []
    boolean_columns: list[str] = []
    _use_columns: list[str] = []

    def __init__(
        self,
        source: Union[Path, pd.DataFrame],
        all_columns: Optional[bool] = True,
        required_columns: Optional[list[str]] = None,
        parent: Optional["GtfsDataset"] = None,
    ) -> None:
        """
        Args:
            source: Location where to find data to use.
            all_columns: Whether desired to return all columns when reading data or not.
            required_columns: List of columns required for reading data. If not set, the defaults will be used.
            parent: Parent GTFS dataset object.
        """
        # save variables
        self.all_columns = all_columns
        self.parent = parent

        # if required columns provided, overwrite defaults
        if required_columns is not None:
            self.required_columns = required_columns

        # make sure the listed path is a path
        if isinstance(source, str):
            self._df = None
            self.file_path = Path(source)
        elif isinstance(source, Path):
            self._df = None
            self.file_path = source
        else:  # data frame
            self._df = source
            self.file_path = None

        # if no columns listed to use, just use required columns if populated
        if len(self._use_columns) == 0 and len(self.required_columns) > 0:
            self._use_columns = self.required_columns

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.file_path}"

    @cached_property
    def use_columns(self) -> list[str]:
        """Columns to be used when reading data if ``all_columns`` is ``False``."""
        if len(self._use_columns) == 0 and len(self.required_columns) > 0:
            cols = self.required_columns
        else:
            cols = self._use_columns

        return cols

    @cached_property
    def file_columns(self) -> list[str]:
        """List of column names observed in the source."""
        # if no file, constructed directly from data frame
        if self.file_path is None:
            df = self._df

        else:
            # only read the first five columns to sniff the data
            df = pd.read_csv(
                self.file_path, header=0, nrows=5, encoding_errors="ignore"
            )

        # get the columns
        columns = df.columns.tolist()

        return columns

    def _ensure_columns(self) -> None:
        """Ensure required columns are in the source data."""
        missing_cols = [c for c in self.required_columns if c not in self.file_columns]
        if len(missing_cols):
            raise MissingRequiredColumnError(
                f"{self.__class__.__name__} required columns missing {missing_cols} when reading file from "
                f"{self.file_path}"
            )

    @cached_property
    def use_columns(self) -> list[str]:
        """
        List of columns used to read the data from the file.

        .. note::
            Columns will *only* be returned if they are detected in the source data.
        """
        # ensure columns in _use_columns are in the required data
        delta_cols = [c for c in self.required_columns if c not in self._use_columns]

        # tack any required columns not listed in use columns onto the end of the list
        if len(delta_cols):
            cols = self._use_columns + delta_cols
        else:
            cols = self._use_columns

        # ensure the columns are in the source data
        cols = [c for c in cols if c in self.file_columns]

        return cols

    @cached_property
    def pandas_dtype(self):
        """Dictionary used to set data types on columns being read in."""
        # create the dictionary to start populating
        dtype = {}

        # for each column in the source data, only add the column to dtype if present in source
        for col in self.file_columns:
            # ensure string columns are encoded as such
            if col in self.string_columns:
                dtype[col] = "string"

            # ensure integer columns are correctly cast to int64 so can handle nulls
            elif col in self.integer_columns or col in self.boolean_columns:
                dtype[col] = "Int64"

            # ensure float columns are also encoded as such
            elif col in self.float_columns:
                dtype[col] = "Float64"

            # if an id column not explicitly cast above, ensure is a string
            elif col.endswith("_id"):
                dtype[col] = "string"

        return dtype

    def _read_source(self, all_columns: bool) -> pd.DataFrame():
        """Helper to read the CSV file."""
        # if the data frame is not already populated, read it from the source file.
        if self._df is None:
            # check to see if the file even exists
            if not self.file_path.exists():
                raise FileNotFoundError(
                    f"Cannot locate {self.file_path} to create the data for {self.__class__.__name__}."
                )

            # ensure required columns are in source data
            self._ensure_columns()

            # if reading all the columns, read the data and return the result
            if all_columns:
                df = pd.read_csv(
                    filepath_or_buffer=self.file_path,
                    sep=",",
                    header=0,
                    dtype=self.pandas_dtype,
                    encoding_errors="ignore",
                    low_memory=False,
                )

            # otherwise, just return the columns listed to use
            else:
                df = pd.read_csv(
                    filepath_or_buffer=self.file_path,
                    usecols=self.use_columns,
                    sep=",",
                    header=0,
                    dtype=self.pandas_dtype,
                    encoding_errors="ignore",
                    low_memory=False,
                )

            # make sure zero length or all space strings are null
            df = replace_zero_and_space_strings_with_nulls(df)

        else:
            # make sure columns are present
            self._ensure_columns()

            # use cached data
            df = self._df

            # prune schema if necessary
            if not all_columns and df.columns.tolist() != self.use_columns:
                df = df.loc[:, self.use_columns]

        # handle if file path does not exist (possible if inferring calendar)
        itm_nm = (
            f"{self.__class__.__name__}"
            if self.file_path is None
            else self.file_path.stem
        )

        logging.info(f"Raw {itm_nm} record count: {df.shape[0]:,}")

        return df

    def _read_source_dask(
        self, all_columns: Optional[bool] = False, usecols: Optional[list[str]] = None
    ) -> pd.DataFrame():
        """Helper to read the CSV file using dask."""
        # check to see if the file even exists
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Cannot locate {self.file_path} to create the data for {self.__class__.__name__}."
            )

        # ensure required columns are in source data
        self._ensure_columns()

        # if reading all the columns, read the data and return the result
        if all_columns or self.use_columns is None:
            ddf = dd.read_csv(
                urlpath=self.file_path,
                sep=",",
                header=0,
                dtype=self.pandas_dtype,
            )

        # if overriding columns to use
        elif usecols is not None:
            ddf = dd.read_csv(
                urlpath=self.file_path,
                usecols=usecols,
                sep=",",
                header=0,
                dtype=self.pandas_dtype,
            )

        # otherwise, just return the columns listed to use
        else:
            ddf = dd.read_csv(
                urlpath=self.file_path,
                usecols=self.use_columns,
                sep=",",
                header=0,
                dtype=self.pandas_dtype,
            )

        # make sure zero length or all space strings are null
        ddf = replace_zero_and_space_strings_with_nulls(ddf)

        logging.info(f"Raw {self.file_path.stem} record count: {len(ddf.index):,}")

        return ddf

    # @cached_property
    # def dask_data(self) -> dd.DataFrame:
    #     """Dask dataframe of the file data."""
    #     df = self._read_source_dask(self.all_columns)
    #     return df

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""
        df = self._read_source(self.all_columns)
        return df

    @cached_property
    def count(self) -> int:
        """Record count in the data frame."""
        return self.data.shape[0]


class GtfsAgency(GtfsFile):
    """Agency GTFS file."""

    string_columns = ["agency_id", "agency_name"]
    _use_columns = ["agency_id", "agency_name"]

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""

        # get the data frame
        df = self._read_source(self.all_columns)

        # if the agency_id is not in the source data, add one to use
        if "agency_id" not in df.columns:
            df["agency_id"] = "1"

        # if the agency name is not in the table, use the value from the ID column
        if "agency_name" not in df.columns:
            df["agency_name"] = df["agency_id"]

        # create an agency UID to distinguish agencies

        return df

    # TODO: add uid from transit index


class GtfsCalendar(GtfsFile):
    """Calendar GTFS file."""

    required_columns = [
        "service_id",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "start_date",
        "end_date",
    ]
    string_columns = ["service_id", "start_date", "end_date"]
    integer_columns = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""

        # get the data frame
        df = self._read_source(self.all_columns)

        # convert the start and end dates to datetime
        for dt_col in ("start_date", "end_date"):
            df[dt_col] = pd.to_datetime(df[dt_col], format="%Y%m%d")

        return df


class GtfsCalendarDates(GtfsFile):
    """Calendar dates GTFS file."""

    required_columns = ["service_id", "date", "exception_type"]
    string_columns = ["service_id"]
    integer_columns = ["exception_type"]

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""

        # get the data frame
        df = self._read_source(self.all_columns)

        # convert date column to datetime
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

        return df


class GtfsFrequencies(GtfsFile):
    """Frequencies GTFS file."""

    required_columns = [
        "trip_id",
        "start_time",
        "end_time",
        "headway_secs",
    ]
    string_columns = ["trip_id", "start_time", "end_time"]
    integer_columns = ["headway_secs"]
    boolean_columns = ["exact_times"]

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""

        # get the data frame
        df = self._read_source(self.all_columns)

        # cast arrival and departure times to timedelta objects
        for col in ["start_time", "end_time"]:
            df[col] = df[col].apply(lambda val: hh_mm_to_timedelta(val))

        return df


class GtfsRoutes(GtfsFile):
    """Routes GTFS file."""

    required_columns = ["route_id", "route_type"]
    string_columns = ["route_id", "agency_id", "route_type"]
    _use_columns = ["route_id", "route_type", "agency_id"]

    def __init__(
        self,
        source: Path,
        all_columns: Optional[bool] = True,
        standardize_route_types: Optional[bool] = False,
        parent: Optional["GtfsDataset"] = None,
    ) -> None:
        """
        Args:
            source: Location where to find file to read.
            all_columns: Whether desired to return all columns when reading data or not.
            standardize_route_types: Whether to standardize route types from any potential European transit route
                types to standard GTFS route types.
        """
        super().__init__(source, all_columns=all_columns, parent=parent)
        self.standardize_route_types = standardize_route_types

    @cached_property
    def _raw_df(self) -> pd.DataFrame:
        return self._read_source(self.all_columns)

    def validate(self, enforce_gtfs_strict: Optional[bool] = False):
        """
        Run validation on routes data.

        * ensure modality codes (``route_type``) are valid values

        Args:
            enforce_gtfs_strict: Whether to use the strict interpretation of modality codes defined in the GTFS
                specification, or whether to allow the expanded European codes. The default is ``False``, to allow the
                European codes as well.

        """
        valid = validate_modality_codes(
            self._raw_df,
            modality_codes_column="route_type",
            enforce_gtfs_strict=enforce_gtfs_strict,
        )

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""
        # validate the input data
        self.validate(enforce_gtfs_strict=False)

        # get the data frame
        df = self._raw_df

        # if the agency id column is not in the input, add it
        if "agency_id" not in df.columns:
            df["agency_id"] = pd.Series(dtype=str)

        # if the agency_id is null, try to populate from the agency table...possible if only one agency
        if isinstance(self.parent, GtfsDataset):
            if self.parent.agency.data.shape[0] == 1:
                df["agency_id"] = self.parent.agency.data.loc[0, "agency_id"]

        # make sure the route types table is correctly coded
        df["route_type"] = (
            df["route_type"].astype("float").astype("int64").astype("str")
        )

        # if desired to standardize the route types, do so
        if self.standardize_route_types:
            df = std_rt_typs(df, route_type_column="route_type")

        return df

    @cached_property
    def sedf(self) -> pd.DataFrame:
        """
        Routes with geometry from shapes as a Spatially Enabled Data Frame.
        """
        if self.parent is None:
            raise ValueError(
                "Cannot detect parent GtfsDataset, so cannot retrieve shapes."
            )

        elif not self.parent.shapes.file_path.exists():
            raise FileNotFoundError(
                "Cannot locate the shapes.txt file. Consequently, cannot create spatially enabled data frame for "
                f"routes in {self.parent.gtfs_folder}"
            )

        else:
            df = (
                self.data.join(
                    self.parent._crosstab_shape_route.set_index("route_id"),
                    on="route_id",
                    how="left",
                )
                .sort_values("shape_id", na_position="last")
                .drop_duplicates("route_id", keep="first")
                .join(
                    self.parent.shapes.sedf.set_index("shape_id"),
                    on="shape_id",
                    how="left",
                )
                .reset_index(drop=True)
                .spatial.set_geometry("SHAPE", inplace=False)
            )
        return df

    def get_temporal_window(
        self,
        start_hour: Optional[int] = 10,
        end_hour: Optional[int] = 4,
        trip_count_column: Optional[str] = "late_night_trip_count",
    ) -> pd.DataFrame:
        """
        Get status of route providing service between a start and end time.

        Args:
            start_hour: Start hour of late night. Default is 22 (10pm).
            end_hour: End hour of late night. Default is 4.
            trip_count_column: Name for output trip count column. Default is ``late_night_trip_count``.
        """
        # get late night data
        ln_trp_srs = self.parent.trips.get_temporal_window(
            start_hour=start_hour, end_hour=end_hour
        )
        ln_trp_srs.name = trip_count_column

        # get the late night trip count by joining trip overnight service with routes
        ln_rt_df = (
            self.parent._crosstab_trip_route.join(
                ln_trp_srs.astype("int64"), on="trip_id"
            )
            .drop(columns="trip_id")
            .groupby("route_id")
            .sum()
        )

        return ln_rt_df

    @cached_property
    def late_night(self) -> pd.DataFrame:
        """Get status of trip providing late night service, defined as service between 10pm and 4am."""
        return self.get_temporal_window()

    @cached_property
    def off_hours(self) -> pd.DataFrame:
        """Get status of trip providing off-hours service, defined as service between 3am and 5am."""
        return self.get_temporal_window(
            start_hour=3, end_hour=5, trip_count_column="off_hours_trip_count"
        )

    @cached_property
    def weekend_service(self) -> pd.DataFrame:
        """Get the trip count offered on weekends for the route."""
        wknd_df = (
            self.parent.trips.data[["route_id", "trip_id"]]
            .join(self.parent.trips.weekend_service, on="trip_id", how="left")
            .drop(columns="trip_id")
            .groupby("route_id")
            .sum()
            .rename(
                columns={
                    "weekend_day_count": "weekend_trip_count",
                    "saturday": "saturday_trip_count",
                    "sunday": "sunday_trip_count",
                }
            )
        )

        return wknd_df


class GtfsShapes(GtfsFile):
    """Shapes GTFS file."""

    required_columns = ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    string_columns = ["shape_id"]
    integer_columns = ["shape_pt_sequence"]
    float_columns = ["shape_pt_lat", "shape_pt_lon"]

    @cached_property
    def sedf(self) -> pd.DataFrame:
        """Spatially enabled data frame of the shapes data as polylines."""
        # if duckdb is available, use it to distill the table
        if importlib.util.find_spec("duckdb") is not None:
            import duckdb

            # read the raw data into duckdb
            shapes_ddb_raw = duckdb.read_csv(
                self.file_path,
                dtype={
                    "shape_id": "VARCHAR",
                    "shape_pt_lon": "DOUBLE",
                    "shape_pt_lat": "DOUBLE",
                    "shape_pt_sequence": "INT64",
                },
            )

            # combine the cordinates into a single column
            shapes_ddb_coords = duckdb.sql(
                """
                SELECT shape_id, CONCAT('[', shape_pt_lon, ',' , shape_pt_lat, ']') AS coordinates 
                FROM shapes_ddb_raw 
                ORDER BY shape_id, shape_pt_sequence 
            """
            )

            # concatenate the coordinates into a sequence for each shape id (line)
            shapes_ddb_arr = duckdb.sql(
                """
                SELECT shape_id, GROUP_CONCAT(coordinates) AS coord_arr 
                FROM shapes_ddb_coords 
                GROUP BY shape_id
            """
            )

            # add the additional json defining the coordinates as a json geometry object
            shapes_ddb_shp = duckdb.sql(
                """
                SELECT shape_id, CONCAT('{"paths": [[', coord_arr , ']], "spatialReference" : {"wkid" : 4326}}') AS SHAPE 
                FROM shapes_ddb_arr
            """
            )

            # convert to pandas data frame
            sedf = shapes_ddb_shp.df()

            # convert the geometry strings to geometry objects
            sedf["SHAPE"] = sedf["SHAPE"].apply(Polyline)

            # set the geometry so recognized as SeDF
            sedf.spatial.set_geometry("SHAPE", inplace=True)

        # if duckdb not available, use pandas
        else:
            # if there is data to work with
            if self.data.shape[0] > 0:
                sedf = (
                    self.data.sort_values(["shape_id", "shape_pt_sequence"])
                    .loc[:, ["shape_id", "shape_pt_lon", "shape_pt_lat"]]
                    .groupby("shape_id")
                    .apply(lambda r: list(zip(r["shape_pt_lon"], r["shape_pt_lat"])))
                    .apply(
                        lambda coords: Polyline(
                            {"paths": [coords], "spatialReference": {"wkid": 4326}}
                        )
                    )
                    .rename("SHAPE")
                    .to_frame()
                    .reset_index()
                    .spatial.set_geometry("SHAPE", inplace=False)
                )

            else:
                sedf = pd.DataFrame(columns=["shape_id", "SHAPE"])

        return sedf


class GtfsStops(GtfsFile):
    """Stops GTFS file."""

    required_columns = ["stop_lat", "stop_lon", "stop_id"]
    string_columns = [
        "stop_id",
        "parent_station",
        "level_id",
        "platform_code",
        "location_type",
        "wheelchair_boarding",
    ]
    float_columns = ["stop_lat", "stop_lon"]

    def _ensure_parent(self) -> None:
        """Helper to make sure there is a parent"""
        if self.parent is None:
            raise ValueError(
                "A parent GTFS object is required to be able to retrieve stop properties from the routes."
            )

    @cached_property
    def _valid_and_invalid_data(self) -> pd.DataFrame():
        """Helper to read and run data through validation."""
        # get the data frame
        df = self._read_source(self.all_columns)

        # validate to get valid (usable) and invalid rows
        df, invalid_df = validate_stop_rows(df, enforce_gtfs_strict=False, copy=False)

        # provide notification if invalid rows
        if len(invalid_df) > 0:
            logging.warning(
                f"There are {len(invalid_df):,} unusable stops in the input data, {self.file_path}\nThese "
                f"invalid records, along with the reason can be accessed through GtfsDataset.stops.invalid_data"
            )

        return df, invalid_df

    @property
    def invalid_data(self) -> pd.DataFrame:
        """Pandas data frame of unusable rows."""
        return self._valid_and_invalid_data[1]

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""
        # get the data frame
        df = self._valid_and_invalid_data[0]

        # add the parent station column and location type columns if not in the source data
        if "parent_station" not in df.columns:
            df["parent_station"] = pd.Series(dtype=str)

        if "location_type" not in df.columns:
            df["location_type"] = pd.Series(dtype=str)

        # apply a default location type if not populated
        df["location_type"].fillna("0", inplace=True)

        # ensure wheelchair boarding column is included
        if "wheelchair_boarding" not in df.columns:
            df["wheelchair_boarding"] = pd.Series(dtype=str)

        # provide default for consistency
        df["wheelchair_boarding"] = df["wheelchair_boarding"].fillna("0")

        # if duplicate stop ids exist, cannot assign wheelchair accessibility from parent to child stops
        if df["stop_id"].duplicated().any():
            logging.warning(
                "Duplicate stop IDs detected. Cannot assign wheelchair accessibility from parent to child stops."
            )

        # if no duplicates, can assign wheelchair accessibility from parents to children
        else:
            # get wheelchair access with stop_id index where wheelchair accessibility is not null or zero
            whlchr_srs = df.loc[
                df["wheelchair_boarding"] != "0", ["stop_id", "wheelchair_boarding"]
            ].set_index("stop_id")["wheelchair_boarding"]

            # filter to just eligible children, stops (location type 0 or null) and entrance/exit (location type 2)
            # and those without wheelchair boarding
            chld_fltr = (
                (df["location_type"] == "0") | (df["location_type"] == "2")
            ) & (df["wheelchair_boarding"] == "0")

            # create a series of the wheelchair accessibility for the parent stations
            whlchr_prnt_df = df.loc[chld_fltr, ["stop_id", "parent_station"]].join(
                whlchr_srs, on="parent_station", how="left"
            )
            whlchr_prnt = whlchr_prnt_df.set_index("stop_id")["wheelchair_boarding"]

            # fill wheelchair accessible values in the child station with those from the parent station if wheelchair
            # accessibility is not populated
            df.loc[chld_fltr, "wheelchair_boarding"] = (
                df.loc[chld_fltr, ["stop_id", "parent_station"]]
                .join(whlchr_prnt, on="parent_station", how="left")
                .set_index("stop_id")["wheelchair_boarding"]
            )

        return df

    @cached_property
    def sedf(self) -> pd.DataFrame:
        """Spatially enabled data frame of the stops data as points."""
        # get the data frame of the data
        df = self.data.copy()

        # create geometry from the longitude (X) and latitude (Y) columns
        df["SHAPE"] = df[["stop_lon", "stop_lat"]].apply(
            lambda r: Point({"x": r[0], "y": r[1], "spatialReference": {"wkid": 4326}}),
            axis=1,
        )
        df.spatial.set_geometry("SHAPE", inplace=True)

        return df

    @cached_property
    def service_days(self):
        """Days of the week with service."""
        df_svc = (
            self.parent._crosstab_stop_service.merge(
                self.parent.calendar.data,
                on="service_id",
                how="outer",
            )
            .loc[
                :,
                [
                    "stop_id",
                    "monday",
                    "tuesday",
                    "wednesday",
                    "thursday",
                    "friday",
                    "saturday",
                    "sunday",
                ],
            ]
            .groupby("stop_id")
            .any()
            .astype("Int64")
        )
        return df_svc

    def _retrieve_route_column(self, column_name: str) -> pd.DataFrame:
        """Helper function to retrieve a data frame of a specific property from the route table."""
        # make sure there is a parent
        self._ensure_parent()

        # make sure the column is in the route table
        if column_name not in self.parent.routes.data.columns:
            raise ValueError(
                f"{column_name} column is not present in the routes table."
            )

        # get a data frame of parent identifier, child identifier, and child column to return
        prnt_chld_df = (
            self.data[["parent_station", "stop_id"]]
            .rename(columns={"parent_station": "parent_id"})
            .merge(self.parent._crosstab_stop_route, on="stop_id", how="left")
            .merge(
                self.parent.routes.data[["route_id", column_name]],
                on="route_id",
                how="left",
            )
            .drop(columns=["route_id"])
        )

        # get child column as a comma separated string by parent identifier
        parent_col_df = (
            prnt_chld_df[~prnt_chld_df["parent_id"].isnull()]
            .groupby("parent_id")[column_name]
            .agg(
                lambda srs: np.NaN
                if srs.isnull().all()
                else ",".join(srs[~srs.isnull()].astype(str).drop_duplicates())
            )
            .dropna()
            .to_frame()
        )

        # start creating a type data frame starting with the parent column values'
        col_df = prnt_chld_df.join(
            parent_col_df, on="stop_id", lsuffix="_orig", how="left"
        )

        # populate non-parent stop column values, these are the stops with child column originally populated
        orig_fltr = col_df[column_name].isnull()
        col_df.loc[orig_fltr, column_name] = col_df.loc[
            orig_fltr, f"{column_name}_orig"
        ]

        # now, add a colum with the parent values, values from the parent stop
        col_df = col_df.join(
            col_df[["stop_id", column_name]].set_index("stop_id"),
            on="parent_id",
            rsuffix="_parent",
            how="left",
        )

        # use parent values to fill any records not already populated
        child_fltr = col_df[column_name].isnull()
        col_df.loc[child_fltr, column_name] = col_df.loc[
            child_fltr, f"{column_name}_parent"
        ]

        # prune the schema to just needed columns
        col_df = col_df.loc[:, ["stop_id", column_name]]

        # clean up the data frame
        col_df = col_df.drop_duplicates().set_index("stop_id")

        return col_df

    @cached_property
    def modalities(self) -> pd.DataFrame:
        """
        Stop modality (``route_type``) for every stop.
        """
        # get a data frame with stop id, route id, and route type (codes)
        stp_typ_df = self.parent.lookup_stop_route.merge(
            self.parent.routes.data[["route_id", "route_type"]],
            on="route_id",
            how="left",
        )

        # create a series based on the
        stp_typ = (
            stp_typ_df[["stop_id", "route_type"]]
            .groupby("stop_id")["route_type"]
            .apply(lambda val: ",".join(sorted(set(val.dropna().astype(str)))))
        )

        return stp_typ

    @cached_property
    def agency(self) -> pd.DataFrame:
        """
        Get stop agency information for stops. This is a three-step process to get agency data for as many
        stops as possible.

        Stop agency data is looked up by retrieving the data from ``agency.txt``. Looking these up requires
        traversing ``stops > stop_times > trips > routes > agency``. Because of this, parent stations, since they do
        not have stop times (stop times are assigned to the child stops), the parent stations will not have a  modality.
        The same is true for many child stops in large train stations, the stops for platforms. After filling as many
        null ``route_type`` rows as possible by retrieving parent agency data, remaining nulls are attempted to be
        filled by using agency data from the parent.
        """
        # ensure parent exists
        self._ensure_parent()

        # if there is only one agency, no need to do complicated lookup
        if len(self.parent.agency.data.index) == 1:
            df = self.data["stop_id"].to_frame()
            df["agency_id"] = self.parent.agency.data.loc[0, "agency_id"]

        # otherwise, do the legwork to lookup through the schema...lots of crosstabs lookups
        else:
            # df = self._retrieve_route_column("agency_id")
            df = self.parent.lookup_stop_route.merge(
                self.parent.routes.data[["route_id", "agency_id"]],
                on="route_id",
                how="left",
            ).drop(columns=["route_id"])

        return df

    @cached_property
    def data_with_agency(self) -> pd.DataFrame:
        """
        Stops data frame with agency id and name. Since multiple agencies can serve a single stop, stops may be
        listed more than once.
        """
        df = self.data.join(self.agency, on="stop_id", how="left")
        return df

    @cached_property
    def sedf_with_agency(self) -> pd.DataFrame:
        """
        Stops spatially enabled data frame with agency id and name. Since multiple agencies can serve a single stop,
        stops may be listed more than once.
        """
        # add the agency to the data
        df = self.data_with_agency

        # create geometry from the longitude (X) and latitude (Y) columns
        df["SHAPE"] = df[["stop_lon", "stop_lat"]].apply(
            lambda r: Point({"x": r[0], "y": r[1], "spatialReference": {"wkid": 4326}}),
            axis=1,
        )
        df.spatial.set_geometry("SHAPE", inplace=True)

        return df

    @cached_property
    def headway_stats(self) -> pd.DataFrame:
        """
        Headway descriptive statistics for each stop, useful for understanding headway as a quality-of-service
        metric for each stop.

        .. note::

            If wanting to access a specific statistic, it is *much* more efficient to directly calculate the
            specific statistic from ``stop_times``. For instance, if you want the average (mean) headway for each
            stop, it is much more efficient to use ``gtfs.stop_times.headway.groupby("stop_id").mean()`` than
            to use ``gtfs.stops.headway_stats["mean"]``.

        """
        hw_stats_df = self.parent.stop_times.headway_stats
        return hw_stats_df

    @cached_property
    def trip_count(self) -> pd.DataFrame:
        """Weekly trip count for each stop."""
        cnt_df = (
            self.parent.lookup_stop_trip.merge(
                self.parent.trips.service_days, on="trip_id"
            )[["stop_id", "service_days"]]
            .groupby("stop_id")
            .sum()
            .rename(columns={"service_days": "trip_count"})
        )
        return cnt_df

    @cached_property
    def route_count(self) -> pd.DataFrame:
        """Weekly route count for each stop."""
        cnt_df = (
            self.parent._crosstab_stop_route.groupby("stop_id")
            .count()
            .rename(columns={"route_id": "route_count"})
        )
        return cnt_df


class GtfsStopTimes(GtfsFile):
    """Stop times GTFS file."""

    required_columns = [
        "trip_id",
        "arrival_time",
        "departure_time",
        "stop_id",
        "stop_sequence",
    ]
    string_columns = ["trip_id", "stop_id", "arrival_time", "departure_time"]
    integer_columns = ["stop_sequence"]

    def __init__(
        self,
        source: Path,
        all_columns: Optional[bool] = True,
        infer_missing: Optional[bool] = False,
        parent: Optional["GtfsDataset"] = None,
    ) -> None:
        """
        Args:
            source: Location where to find file to read.
            all_columns: Whether desired to return all columns when reading data or not.
            infer_missing: Whether to infer missing arrival and departure values when reading data or not.
        """
        super().__init__(source, all_columns=all_columns, parent=parent)
        self.infer_missing = infer_missing

    @cached_property
    def _raw_data(self) -> pd.DataFrame:
        """Raw data without interpolated values...faster for crosstabs."""
        # get the data frame
        df = self._read_source(self.all_columns)

        # cast arrival and departure times to timedelta objects
        for col in ["arrival_time", "departure_time"]:
            df[col] = df[col].apply(lambda val: hh_mm_to_timedelta(val))

        return df

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""

        df = self._raw_data

        # interpolate any missing stop times if desired
        if self.infer_missing:
            df = interpolate_stop_times(df)

        return df

    @cached_property
    def headway(self) -> pd.DataFrame:
        """
        Data frame with ``stop_id`` and ``headway`` for each stop. This is extremely useful for deriving descriptive
        statistics about each stop.

        .. code-block:: python

            from pathlib import Path

            from gtfs_tools.gtfs import GTFS

            # path to data
            gtfs_dir = Path("C:/data/gtfs/gtfs_agency_dataset")

            # create GTFS object instance
            gtfs = GTFS(gtfs_dir)

            # get headway values for stops
            headway_df = gtfs.stops.headway

            # calculate headway descriptive statistics for each stop
            headway_stats = headway_df.groupby("stop_id").describe()

        """
        headway_df = calculate_headway(self.data)

        return headway_df

    @cached_property
    def headway_stats(self) -> pd.DataFrame:
        """Utility property to quickly get headway descriptive statistics by stop."""
        headway_stats_df = self.headway.groupby("stop_id").describe()["headway"]
        return headway_stats_df


class GtfsTrips(GtfsFile):
    """Trips GTFS file."""

    required_columns = ["trip_id", "route_id", "service_id"]
    string_columns = [
        "trip_id",
        "route_id",
        "trip_short_name",
        "service_id",
        "shape_id",
    ]
    integer_columns = ["wheelchair_accessible", "bikes_allowed"]
    _use_columns = [
        "trip_id",
        "route_id",
        "service_id",
        "wheelchair_accessible",
        "bikes_allowed",
    ]

    @cached_property
    def sedf(self) -> pd.DataFrame:
        if self.parent is None:
            raise ValueError(
                "Cannot detect parent GtfsDataset, so cannot retrieve shapes for the Spatially Enabled Dataframe."
            )

        else:
            df = (
                self.data.join(
                    self.parent.shapes.sedf.set_index("shape_id"),
                    on="shape_id",
                    how="left",
                )
                .reset_index(drop=True)
                .spatial.set_geometry("SHAPE", inplace=False)
            )
        return df

    @cached_property
    def service_days(self) -> pd.Series:
        """Days of the week the trip offers service."""
        # day of week columns in the calendar file
        dow_cols = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]

        # although not supposed to happen, catch potential
        svc = self.parent.calendar.data.set_index("service_id")[dow_cols].apply(
            lambda r: sum(r), axis=1
        )
        svc.name = "service_days"

        # attach the service days to trips based on service id, and remove trip id
        svc_days = (
            self.data.loc[:, ["trip_id", "service_id"]]
            .join(svc, on="service_id")
            .drop(columns="service_id")
            .set_index("trip_id")
        )

        return svc_days

    def get_temporal_window(
        self,
        column_name: str,
        start_hour: Optional[int] = 10,
        start_minutes: Optional[int] = 0,
        end_hour: Optional[int] = 4,
        end_minutes: Optional[int] = 0,
    ) -> pd.DataFrame:
        """
        Get status of trip providing service between start and end times.

        Args:
            column_name: Name of column to add providing temporal status status for.
            start_hour: Start hour of temporal window. Default is 22 (10pm).
            end_minutes: Start minutes of temporal window. Default is 0.
            end_hour: End hour of temporal window. Default is 4.
            start_minutes: End minutes of temporal window. Default is 0.
        """
        # prune schema to just columns needed for analysis
        st_df = self.parent.stop_times.data.loc[:, ["trip_id", "arrival_time"]]

        # calculate decimal hours
        start_time = start_hour + start_minutes / 60
        end_time = end_hour + end_minutes / 60

        # handle if spanning midnight (start hour is greater than end hour)
        if start_time < end_time:
            # calculate if after start time
            st_df["after_start_time"] = (
                st_df["arrival_time"].dt.total_seconds() / 3600 % 24 > start_time
            )

            # calculate if before end time
            st_df["before_end_time"] = (
                st_df["arrival_time"].dt.total_seconds() / 3600 % 24 < end_time
            )

            # consolidate by trip_id to get any of the stop times respectively after and before the cutoffs
            ln_df = st_df.groupby("trip_id")[
                ["after_start_time", "before_end_time"]
            ].any()

            # if after start time and before the end time
            ln_srs = ln_df["after_start_time"] & ln_df["before_end_time"]

            # add the column name
            ln_srs.name = column_name

        # handle normal pattern, start time less than end time
        else:
            # calculate, by stop, if providing late night service, using modulus to account for trips spanning midnight
            st_df[column_name] = (
                st_df["arrival_time"].dt.total_seconds() / 3600 % 24 > start_time
            ) | (st_df["arrival_time"].dt.total_seconds() / 3600 % 24 < end_time)

            # consolidate by trip_id
            ln_srs = st_df.groupby("trip_id")[column_name].any()

        # convert to dataframe for consistency
        ln_df = ln_srs.to_frame()

        return ln_df

    @cached_property
    def early_morning(self) -> pd.DataFrame:
        """Trips providing early morning service (3am to 6am)."""
        df = self.get_temporal_window(
            column_name="early_morning",
            start_hour=3,
            end_hour=6,
        )
        return df

    @cached_property
    def morning_peak(self) -> pd.DataFrame:
        """Trips providing morning peak, "rush hour," service (6am to 9am)."""
        df = self.get_temporal_window(
            column_name="morning_peak",
            start_hour=6,
            end_hour=9,
        )
        return df

    @cached_property
    def midday(self) -> pd.DataFrame:
        """Trips providing midday service (9am to 3pm)."""
        df = self.get_temporal_window(
            column_name="midday",
            start_hour=9,
            end_hour=15,
        )
        return df

    @cached_property
    def afternoon_peak(self) -> pd.DataFrame:
        """Trips providing afternoon peak, "rush hour," service (3pm to 6pm)."""
        df = self.get_temporal_window(
            column_name="afternoon_peak",
            start_hour=15,
            end_hour=18,
        )
        return df

    @cached_property
    def evening(self) -> pd.DataFrame:
        """Trips providing evening service (6am to 9am)."""
        df = self.get_temporal_window(
            column_name="evening",
            start_hour=18,
            end_hour=22,
        )
        return df

    @cached_property
    def late_night(self) -> pd.DataFrame:
        """Get status of trip providing late night service, defined as service between 10pm and 4am."""
        df = self.get_temporal_window(
            column_name="late_night",
            start_hour=22,
            end_hour=3,
        )
        return df

    @cached_property
    def temporal_windows(self) -> pd.DataFrame:
        """
        Get status of trip providing service in the following temporal windows:
        * Early Morning: 3am to 6am
        * Morning Peak: 6am to 9am
        * Midday: 9am to 3pm
        * Afternoon Peak: 3pm to 6pm
        * Evening: 3pm to 10pm
        * Late Night: 10pm to 3am
        """
        df = (
            self.early_morning.join(self.morning_peak)
            .join(self.midday)
            .join(self.afternoon_peak)
            .join(self.evening)
            .join(self.late_night)
        )
        return df

    @cached_property
    def weekend_service(self) -> pd.DataFrame:
        """
        Data frame indicating if trip provides service on Saturday, Sunday and the count of days weekend service is
        provided.
        """
        # combine weekend days' service columns with the trip id
        wknd_df = self.parent._crosstab_trip_service.merge(
            self.parent.calendar.data.loc[:, ["service_id", "saturday", "sunday"]],
            on="service_id",
        )

        # calculate the count of weekend days providing service
        wknd_df["weekend_day_count"] = wknd_df["saturday"] + wknd_df["sunday"]
        wknd_df = wknd_df.drop(columns="service_id").set_index("trip_id")

        return wknd_df


class GtfsDataset(object):
    """
    Object with ingestion and processing capabilities for working with a GTFS dataset, the starting point for working
    with GTFS data.

    .. code-block:: python

        import pathlib as Path
        from gtfs_tools.gtfs import GtfsDataset

        # path to directory with a gtfs dataset
        gtfs_pth = Path(r"D:\\data\\agency_gtfs")

        # create the GtfsDataset instance
        gtfs = GtfsDataset(gtfs_pth)

        # ensure all required files are included
        gtfs.validate()

        # get a spatially enabled dataframe of all stops with the agency added
        stops_df = gtfs.stops.sedf_with_agency

        # get a spatially enabled dataframe of all routes
        routes_df = gtfs.routes.sedf
    """

    def __init__(
        self,
        gtfs_path: Path,
        infer_stop_times: Optional[bool] = True,
        infer_calendar: Optional[bool] = True,
        required_files: Optional[list[str]] = None,
        standardize_route_types: Optional[bool] = False,
    ) -> None:
        """
        Args:
            gtfs_path: Directory containing GTFS data.
            infer_stop_times: Whether to infer stop times, missing arrival and departure times.
            infer_calendar: Whether to infer calendar from calendar dates if calendar.txt is missing.
            required_files: List of files required for the GTFS dataset. By default, these include ``[ "agency.txt",
                "calendar.txt", "routes.txt", "shapes.txt", "stops.txt",  "stop_times.txt", "trips.txt"]``
            standardize_route_types: Whether to standardize route types from any potential European transit route
                types to standard GTFS route types.
        """
        # ensure the directory is a path
        if isinstance(gtfs_path, str):
            gtfs_path = Path(gtfs_path)

        # if the folder is a path to a zipped archive, extract it to a temp directory
        if gtfs_path.suffix == ".zip":
            gtfs_path = self.unzip(gtfs_path, Path(tempfile.mkdtemp()))

        # save parameters as properties
        self.gtfs_folder = gtfs_path
        self.infer_stop_times = infer_stop_times
        self.infer_calendar = infer_calendar
        self.standardize_route_types = standardize_route_types

        if required_files is None:
            self.required_files = [
                "agency.txt",
                "calendar.txt",
                # "calendar_dates.txt",
                "routes.txt",
                "shapes.txt",
                "stops.txt",
                "stop_times.txt",
                "trips.txt",
            ]
        else:
            self.required_files = required_files

        # ensure required files are present
        valid = self.validate(self.infer_calendar)

        # update gtfs directory location based on location of agency.txt - is there since validate is true
        # this provides support for nested directories inside zipped archives
        if valid:
            self.gtfs_folder = list(self.gtfs_folder.glob("**/agency.txt"))[0].parent

        # paths to child resources
        self._agency_pth = self.gtfs_folder / "agency.txt"
        self._calendar_pth = self.gtfs_folder / "calendar.txt"
        self._calendar_dates_pth = self.gtfs_folder / "calendar_dates.txt"
        self._frequencies_pth = self.gtfs_folder / "frequencies.txt"
        self._routes_pth = self.gtfs_folder / "routes.txt"
        self._shapes_pth = self.gtfs_folder / "shapes.txt"
        self._stops_pth = self.gtfs_folder / "stops.txt"
        self._stop_times_pth = self.gtfs_folder / "stop_times.txt"
        self._trips_pth = self.gtfs_folder / "trips.txt"

    def __str__(self):
        return str(self.gtfs_folder)

    def __repr__(self):
        return f"GtfsDataset: {self.gtfs_folder}"

    @cached_property
    def agency(self) -> GtfsAgency:
        """Agency data from GTFS file."""
        agency = GtfsAgency(self._agency_pth, parent=self)
        return agency

    @cached_property
    def calendar(self) -> GtfsCalendar:
        """
        Calendar data from GTFS file.

        .. note::

            If the ``calendar.txt`` file is not present, by default, this will be inferred from the
            ``calendar-dates.txt`` file.

        """
        # if calendar.txt is present in this dataset
        if self._calendar_pth.exists():
            calendar = GtfsCalendar(self._calendar_pth, parent=self)

        # if calendar.txt does not exist, infer from calendar-dates.txt if desired
        elif self.infer_calendar and self._calendar_dates_pth.exists():
            logging.warning(
                "calendar.txt does not exist, so inferring calendar from calendar-dates.txt"
            )

            # get a raw calendar data frame built from calendar-dates.txt
            raw_df = get_calendar_from_calendar_dates(self.calendar_dates.data)

            # build calendar from the inferred raw data
            calendar = GtfsCalendar(raw_df, parent=self)

        elif self.infer_calendar and not self._calendar_dates_pth.exists():
            raise FileNotFoundError(
                "Cannot locate a calendar.txt or a calendar-dates.txt file in this GTFS dataset."
            )

        else:
            raise FileNotFoundError(
                "A calendar.txt file does not appear to be included in this GTFS dataset."
            )

        return calendar

    @cached_property
    def calendar_dates(self) -> GtfsCalendarDates:
        """Calendar dates data from GTFS file."""
        calendar_dates = GtfsCalendarDates(self._calendar_dates_pth, parent=self)
        return calendar_dates

    @cached_property
    def frequencies(self) -> GtfsFrequencies:
        """Frequencies data from GTFS file."""
        frequencies = GtfsFrequencies(self._frequencies_pth, parent=self)
        return frequencies

    @cached_property
    def routes(self) -> GtfsRoutes:
        """Routes data from GTFS file."""
        routes = GtfsRoutes(self._routes_pth, parent=self)
        return routes

    @cached_property
    def shapes(self) -> GtfsShapes:
        """Shapes data from GTFS file."""
        shapes = GtfsShapes(self._shapes_pth, parent=self)
        return shapes

    @cached_property
    def stops(self) -> GtfsStops:
        """Stops data from GTFS file."""
        stops = GtfsStops(self._stops_pth, parent=self)
        return stops

    @cached_property
    def stop_times(self) -> GtfsStopTimes:
        """
        Stop times data from GTFS file.

        .. note::

            If ``arrival_times`` and ``departure_times`` are only provided for the beginning and end of a trip,
            intermediate stop times will be inferred by evenly distributing the intervals between the provided
            starting and ending times for the route.

        """
        stop_times = GtfsStopTimes(
            self._stop_times_pth, infer_missing=self.infer_stop_times, parent=self
        )
        return stop_times

    @cached_property
    def trips(self) -> GtfsTrips:
        """Trips data from GTFS file."""
        trips = GtfsTrips(self._trips_pth, parent=self)
        return trips

    @classmethod
    def from_zip(
        cls,
        zip_path: Path,
        output_directory: Optional[Path] = None,
        infer_stop_times: Optional[bool] = True,
        infer_calendar: Optional[bool] = True,
        required_files: Optional[list[str]] = None,
        standardize_route_types: Optional[bool] = False,
    ) -> "GtfsDataset":
        """
        Create a ``GtfsDataset`` from a zip file.e

        Args:
            zip_path: Path to the zip file.
            output_directory: Optional directory to output the dataset to. If not provided, data will be unpacked
                to the temp directory.
            infer_stop_times: Whether to infer stop times, missing arrival and departure times.
            infer_calendar: Whether to infer calendar from calendar dates if calendar.txt is missing.
            required_files: List of files required for the GTFS dataset. By default, these include ``[ "agency.txt",
                "calendar.txt", "routes.txt", "shapes.txt", "stops.txt",  "stop_times.txt", "trips.txt"]``
            standardize_route_types: Whether to standardize route types from any potential European transit route
                types to standard GTFS route types.
        """
        # if no directory provided, just use a temp directory
        if output_directory is None:
            output_directory = tempfile.mkdtemp()

        # unpack the gtfs dataset
        unpack_dir = cls.unzip(zip_path, output_directory)

        # find the gtfs dataset(s) in the extracted location
        gtfs_dir_lst = get_gtfs_directories(unpack_dir)

        # ensure only one GTFS dataset is found
        if len(gtfs_dir_lst) == 0:
            raise FileNotFoundError(
                "Cannot locate a GTFS dataset in the extracted files."
            )
        elif len(gtfs_dir_lst) > 1:
            raise Exception(
                "Detected more than one GTFS dataset in the extracted files."
            )
        else:
            gtfs_dir = gtfs_dir_lst[0]

            logging.info(f"GTFS directory is located at {gtfs_dir}")

        # create the GtfsDataset object instance
        gtfs = cls(
            gtfs_dir,
            infer_stop_times=infer_stop_times,
            infer_calendar=infer_calendar,
            required_files=required_files,
            standardize_route_types=standardize_route_types,
        )

        return gtfs

    @staticmethod
    def unzip(zip_path: Path, output_directory: Path) -> Path:
        """
        Unpack a zipped GTFS dataset.

        Args:
            zip_path: Path to the zip file.
            output_directory: Directory to unpack the output the dataset to.
        """
        # unpack the zipped archive
        with zipfile.ZipFile(zip_path, "r") as zipper:
            zipper.extractall(output_directory)

        logging.info(f"Extracted archive from {zip_path} to {output_directory}")

        # ensure path
        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        return output_directory

    def validate(self, calendar_or_calendar_dates: Optional[bool] = True) -> bool:
        """
        Ensure all necessary files are present. Required files are determined by the ``required_files`` construction
        parameter.

        Args:
            calendar_or_calendar_dates: When scanning for files, accept the dataset if ``calendar_dates.txt`` is
                present, even in the absence of ``calendar.txt``.
        """
        valid = validate_required_files(
            self.gtfs_folder,
            required_files=self.required_files,
            calendar_or_calendar_dates=calendar_or_calendar_dates,
        )

        return valid

    @cached_property
    def valid(self):
        """Alias for ``validate`` accepting default parameters."""
        return self.validate()

    @cached_property
    def _crosstab_stop_trip(self) -> pd.DataFrame:
        """Data frame with crosstabs lookup between stops and trips."""
        # if duckdb is available, use this...way faster
        if importlib.util.find_spec("duckdb") is not None:
            # import duckdb
            import duckdb

            # create a duckdb table to read the data
            stop_times_ddb = duckdb.read_csv(
                self.stop_times.file_path,
                dtype={"stop_id": "VARCHAR", "trip_id": "VARCHAR"},
            )

            # create the crosstabs lookup using duckdb to boil down the stop times table to just a stop to trip lookup
            # stop_times_df = duckdb.sql(f"SELECT DISTINCT stop_id, trip_id FROM stop_times_ddb").df()
            stop_times_df = duckdb.sql(
                f"SELECT stop_id, trip_id FROM stop_times_ddb GROUP BY stop_id, trip_id"
            ).df()

        # otherwise, we know dask is available with any ArcGIS Pro installation
        else:
            # read the stop times into a dask dataframe to get unique permutations of stop and trip id's
            stop_times_df = (
                self.stop_times._read_source_dask(usecols=["stop_id", "trip_id"])
                .drop_duplicates()
                .compute()
            )

        # ensure data types are consistent, string
        for col in stop_times_df.columns:
            if stop_times_df[col].dtype == "float":
                stop_times_df[col] = stop_times_df[col].astype("Int64").astype("string")

            else:
                stop_times_df[col] = stop_times_df[col].astype("string")

        # combine the stop times lookup with the stops to ensure complete stop_id's, even if means null trip_id's
        df = self.stops.data[["stop_id"]].merge(stop_times_df, on="stop_id", how="left")

        return df

    @cached_property
    def _crosstab_trip_route(self) -> pd.DataFrame:
        """Data frame with crosstabs lookup between trips and routes. This is useful when attempting to associate trips
        to routes."""
        df = self.trips.data[["trip_id", "route_id"]]
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    @cached_property
    def _crosstab_trip_service(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between trips and services. This is useful when attempting to associate
        trips to calendar."""
        df = self.trips.data[["trip_id", "service_id"]]
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    @cached_property
    def _crosstab_route_agency(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between routes and agencies."""
        df = self.routes.data[["route_id", "agency_id"]]
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    @cached_property
    def _crosstab_stop_route(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between stops and routes."""
        df = self._crosstab_stop_trip.join(
            self._crosstab_trip_route.set_index("trip_id"), on="trip_id", how="left"
        ).loc[:, ["stop_id", "route_id"]]
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    @cached_property
    def _crosstab_stop_agency(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between trips and agency id."""
        df = self._crosstab_stop_route.merge(
            self.routes.data[["route_id", "agency_id"]],
            on="route_id",
            how="left",
        ).drop(columns="route_id")
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    @cached_property
    def _crosstab_stop_service(self) -> pd.DataFrame:
        """
        Data frame with crosstab lookup between stops and service_id. This is useful when attempting to associate
        between stops and calendar.

        .. note::

            This will include *all* stops, even if there is not an associated trip.
        """
        df = self._crosstab_stop_trip.merge(
            self._crosstab_trip_service, on="trip_id", how="left"
        )[["stop_id", "service_id"]]
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    @cached_property
    def _crosstab_shape_route(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between shapes and routes."""
        df = self.trips.data.loc[
            ~self.trips.data["shape_id"].isnull(), ["shape_id", "route_id"]
        ]
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    @cached_property
    def lookup_stop_trip(self) -> pd.DataFrame:
        """
        Crosstabular lookup from stops to trips, populating ``trip_id`` by first looking up parent values
        from children, and then attempting to populate any remaining missing values from parents.
        """

        logging.info(
            f"Stops with associated trips: {self._crosstab_stop_trip.shape[0]:,}"
        )

        # get a dataframe of trips pulled from children stops for parent stops
        parent_df = (
            self.stops.data[["parent_station", "stop_id"]]
            .dropna()
            .merge(self._crosstab_stop_trip, on="stop_id", how="left")
            .drop(columns="stop_id")
            .rename(columns={"parent_station": "stop_id"})
            .drop_duplicates()
        )

        logging.info(
            f"Parent stops with trips looked up from children: {parent_df.shape[0]:,}"
        )

        # get a dataframe of stops without trips, which can be inherited from parents
        child_df = (
            self._crosstab_stop_trip[self._crosstab_stop_trip["trip_id"].isnull()]
            .drop(columns="trip_id")
            .merge(parent_df, on="stop_id", how="left")
            .sort_values(["stop_id", "trip_id"], na_position="last")
            .drop_duplicates()
        )

        logging.info(
            f"Child stops with trips looked up from parents: {child_df.shape[0]:,}"
        )

        # see if we can pull properties down to grandchildren not populated
        grandchild_df = (
            child_df[child_df["trip_id"].isnull()]
            .drop(columns="trip_id")
            .merge(parent_df, on="stop_id", how="left")
            .sort_values(["stop_id", "trip_id"], na_position="last")
            .drop_duplicates()
        )

        logging.info(
            f"Grandchild stops with trips looked up from grandparents: {grandchild_df.shape[0]:,}"
        )

        # get a dataframe combining the raw crosstabs, parents from children, children from parents and grandchildren
        crosstabs_df = (
            pd.concat([self._crosstab_stop_trip, parent_df, child_df, grandchild_df])
            .sort_values(["stop_id", "trip_id"], na_position="last")
            .drop_duplicates()
            .reset_index(drop=True)
        )

        return crosstabs_df

    @cached_property
    def lookup_stop_route(self) -> pd.DataFrame:
        """
        Crosstabular lookup from stops to routes, populating ``route_id`` by first looking up parent values
        from children, and then attempting to populate any remaining missing values from parents.
        """
        crosstabs_df = (
            self.lookup_stop_trip.merge(
                self._crosstab_trip_route, on="trip_id", how="left"
            )
            .drop(columns="trip_id")
            .drop_duplicates()
            .reset_index(drop=True)
        )

        return crosstabs_df

    @cached_property
    def lookup_stop_agency(self) -> pd.DataFrame:
        """
        Crosstablular lookup from stops to agencies, populating ``agency_id`` by first looking up parent values
        from children, and then attempting to populate any remaining missing values from parents.
        """
        # if there is only one agency, no need to do complicated lookup
        if len(self.agency.data.index) == 1:
            df = self.stops.data["stop_id"].to_frame()
            df["agency_id"] = self.agency.data.loc[0, "agency_id"]

        # otherwise, do the legwork to lookup through the schema...lots of crosstabs lookups
        else:
            df = (
                self.lookup_stop_route.merge(
                    self.routes.data[["route_id", "agency_id"]],
                    on="route_id",
                    how="left",
                )
                .drop(columns=["route_id"])
                .drop_duplicates()
                .reset_index(drop=True)
            )

        return df

    def export(self, gtfs_directory: Union[str, Path]) -> Path:
        """Export standardized files to a directory."""
        # make sure the output directory path is a Path
        if isinstance(gtfs_directory, str):
            gtfs_directory = Path(gtfs_directory)

        # if it does not exist, create it
        gtfs_directory.mkdir(exist_ok=True, parents=True)

        # iteratively export files
        export_lst = [
            "agency",
            "calendar",
            "frequencies",
            "routes",
            "shapes",
            "stops",
            "stop_times",
            "trips",
        ]

        for name in export_lst:
            # create the path to save the data
            asset_pth = gtfs_directory / f"{name}.txt"

            # get the object
            asset = getattr(self, name)

            # export the asset
            asset.data.to_csv(asset_pth, index=False)

        return gtfs_directory
