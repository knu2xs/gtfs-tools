"""
GTFS module documentation.
"""
import logging
import tempfile
import zipfile
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

from arcgis.features import GeoAccessor
from arcgis.geometry import Point, Polyline
import pandas as pd

from .utils.dt import hh_mm_to_timedelta
from .utils.exceptions import MissingRequiredColumnError
from .utils.gtfs import get_calendar_from_calendar_dates, interpolate_stop_times
from .utils.pandas import replace_zero_and_space_strings_with_nulls

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
    ) -> None:
        """
        Args:
            source: Location where to find data to use.
            all_columns: Whether desired to return all columns when reading data or not.
            required_columns: List of columns required for reading data. If not set, the defaults will be used.
        """
        # save variables
        self.all_columns = all_columns

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
            if col in self.string_columns:
                dtype[col] = str

            elif col in self.integer_columns or col in self.boolean_columns:
                dtype[col] = "Int64"

            elif col in self.float_columns:
                dtype[col] = float

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

        return df

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""

        # get the data frame
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
        "exact_times",
    ]
    string_columns = ["trip_id", "start_time", "end_time"]
    integer_columns = ["headway_secs"]
    boolean_columns = ["exact_times"]


class GtfsRoutes(GtfsFile):
    """Routes GTFS file."""

    required_columns = ["route_id", "route_type"]
    string_columns = ["route_id", "agency_id"]
    integer_columns = ["route_type"]
    _use_columns = ["route_id", "route_type", "agency_id"]

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""

        # get the data frame
        df = self._read_source(self.all_columns)

        # if the agency id column is not in the input, add it and leave it empty
        if "agency_id" not in df.columns:
            df["agency_id"] = pd.Series(dtype=str)

        return df


class GtfsShapes(GtfsFile):
    """Shapes GTFS file."""

    required_columns = ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    string_columns = ["shape_id"]
    integer_columns = ["shape_pt_sequence"]
    float_columns = ["shape_pt_lat", "shape_pt_lon"]

    @cached_property
    def sedf(self) -> pd.DataFrame:
        """Spatially enabled data frame of the shapes data as polylines."""
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
            .spatial.set_geometry("SHAPE", inplace=False)
        )

        return sedf


class GtfsStops(GtfsFile):
    """Stops GTFS file."""

    required_columns = ["stop_lat", "stop_lon", "stop_id"]
    string_columns = ["stop_id", "parent_station"]
    integer_columns = ["location_type"]
    float_columns = ["stop_lat", "stop_lon"]

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""

        # get the data frame
        df = self._read_source(self.all_columns)

        # add the parent station column and location type columns if not in the source data
        if "parent_station" not in df.columns:
            df["parent_station"] = pd.Series(dtype=str)

        if "location_type" not in df.columns:
            df["location_type"] = pd.Series(dtype=int)

        # apply a default location type if not populated
        df["location_type"].fillna(0, inplace=True)

        return df

    @cached_property
    def sedf(self) -> pd.DataFrame:
        """Spatially enabled data frame of the stops data as points."""
        # get the data frame of the data
        df = self.data.copy()

        # create geometry from the longitude (X) and latitude (Y) columns
        df["SHAPE"] = df["stop_lon", "stop_lat"].apply(
            lambda r: Point({"x": r[0], "y": r[1], "spatialReference": {"wkid": 4326}}),
            axis=1,
        )
        df.spatial.set_geometry("SHAPE", inplace=True)

        return df


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
    ) -> None:
        """
        Args:
            source: Location where to find file to read.
            all_columns: Whether desired to return all columns when reading data or not.
            infer_missing: Whether to infer missing arrival and departure values when reading data or not.
        """
        super().__init__(source, all_columns)
        self.infer_missing = infer_missing

    @cached_property
    def data(self) -> pd.DataFrame:
        """Pandas data frame of the file data."""

        # get the data frame
        df = self._read_source(self.all_columns)

        # cast arrival and departure times to timedelta objects
        for col in ["arrival_time", "departure_time"]:
            df[col] = df[col].apply(lambda val: hh_mm_to_timedelta(val))

        # interpolate any missing stop times if desired
        if self.infer_missing:
            df = interpolate_stop_times(df)

        return df


class GtfsTrips(GtfsFile):
    """Trips GTFS file."""

    required_columns = ["trip_id", "route_id", "service_id"]
    string_columns = ["trip_id", "route_id", "service_id"]
    integer_columns = ["wheelchair_accessible", "bikes_allowed"]
    _use_columns = [
        "trip_id",
        "route_id",
        "service_id",
        "wheelchair_accessible",
        "bikes_allowed",
    ]


class GtfsDataset(object):
    """
    GTFS dataset with ingestion and processing capabilities.
    """

    def __init__(
        self,
        gtfs_folder: Path,
        infer_stop_times: Optional[bool] = True,
        infer_calendar: Optional[bool] = True,
        required_files: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            gtfs_folder: Directory containing GTFS data.
            infer_stop_times: Whether to infer stop times, missing arrival and departure times.
            infer_calendar: Whether to infer calendar from calendar dates if calendar.txt is missing.
            required_files: List of files required for the GTFS dataset. By default, these include ``[ "agency.txt",
                "calendar.txt", "routes.txt", "stops.txt",  "stop_times.txt", "trips.txt"]``
        """
        # ensure the directory is a path
        if isinstance(gtfs_folder, str):
            gtfs_folder = Path(gtfs_folder)

        # save parameters as properties
        self.gtfs_folder = gtfs_folder
        self.infer_stop_times = infer_stop_times
        self.infer_calendar = infer_calendar

        if required_files is None:
            self.required_files = [
                "agency.txt",
                "calendar.txt",
                "calendar_dates.txt",
                "routes.txt",
                "stops.txt",
                "stop_times.txt",
                "trips.txt",
            ]
        else:
            self.required_files = required_files

        # paths to child resources
        self._agency_pth = self.gtfs_folder / "agency.txt"
        self._calendar_pth = self.gtfs_folder / "calendar.txt"
        self._calendar_dates_pth = self.gtfs_folder / "calendar-dates.txt"
        self._frequencies_pth = self.gtfs_folder / "frequencies.txt"
        self._routes_pth = self.gtfs_folder / "routes.txt"
        self._shapes_pth = self.gtfs_folder / "shapes.txt"
        self._stops_pth = self.gtfs_folder / "stops.txt"
        self._stop_times_pth = self.gtfs_folder / "stop_times.txt"
        self._trips_pth = self.gtfs_folder / "trips.txt"

    @cached_property
    def agency(self) -> GtfsAgency:
        """Agency data from GTFS file."""
        agency = GtfsAgency(self._agency_pth)
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
            calendar = GtfsCalendar(self._calendar_pth)

        # if calendar.txt does not exist, infer from calendar-dates.txt if desired
        elif self.infer_calendar and self._calendar_dates_pth.exists():
            logging.debug(
                "calendar.txt does not exist, so inferring calendar from calendar-dates.txt"
            )

            # get a raw calendar data frame built from calendar-dates.txt
            raw_df = get_calendar_from_calendar_dates(self.calendar_dates.data)

            # build calendar from the inferred raw data
            calendar = GtfsCalendar(raw_df)

        else:
            raise FileNotFoundError(
                "calendar.txt file does not appear to be included in this GTFS dataset."
            )

        return calendar

    @cached_property
    def calendar_dates(self) -> GtfsCalendarDates:
        """Calendar dates data from GTFS file."""
        calendar_dates = GtfsCalendarDates(self._calendar_dates_pth)
        return calendar_dates

    @cached_property
    def frequencies(self) -> GtfsFrequencies:
        """Frequencies data from GTFS file."""
        frequencies = GtfsFrequencies(self._frequencies_pth)
        return frequencies

    @cached_property
    def routes(self) -> GtfsRoutes:
        """Routes data from GTFS file."""
        routes = GtfsRoutes(self._routes_pth)
        return routes

    @cached_property
    def shapes(self) -> GtfsShapes:
        """Shapes data from GTFS file."""
        shapes = GtfsShapes(self._shapes_pth)
        return shapes

    @cached_property
    def stops(self) -> GtfsStops:
        """Stops data from GTFS file."""
        stops = GtfsStops(self._stops_pth)
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
            self._stop_times_pth, infer_missing=self.infer_stop_times
        )
        return stop_times

    @cached_property
    def trips(self) -> GtfsTrips:
        """Trips data from GTFS file."""
        trips = GtfsTrips(self._trips_pth)
        return trips

    @classmethod
    def from_zip(
        cls, zip_path: Path, output_directory: Optional[Path] = None
    ) -> "GtfsDataset":
        """
        Create a ``GtfsDataset`` from a zip file.

        Args:
            zip_path: Path to the zip file.
            output_directory: Optional directory to output the dataset to. If not provided, data will be unpacked
                to the temp directory.
        """
        # if no directory provided, just use a temp directory
        if output_directory is None:
            output_directory = tempfile.mkdtemp()

        # unpack the gtfs dataset
        output_directory = cls.unzip(zip_path, output_directory)

        # create the GtfsDataset object instance
        gtfs = cls(output_directory)

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

        # ensure path
        if isinstance(output_directory, str):
            output_directory = Path(output_directory)

        return output_directory

    def validate_files(self, calendar_or_calendar_dates: Optional[bool] = True) -> bool:
        """
        Ensure all necessary files are present. Required files are determined by the ``required_files`` construction
        parameter.

        Args:
            calendar_or_calendar_dates: When scanning for files, accept the dataset if ``calendar_dates.txt`` is
                present, even in the absence of ``calendar.txt``.
        """
        # pull out the required file names without the extension
        req_lst = [f.rstrip(".txt") for f in self.required_files]

        # get files present in the dataset directory
        file_lst = [f.stem for f in self.gtfs_folder.glob("*.txt")]

        # if handling calendar and calendar dates separately
        if calendar_or_calendar_dates:
            # check for the presence of the calendar or calendar_dates file
            for f in file_lst:
                # if calendar or calendar_dates found
                if f in ["calendar", "calendar_dates"]:
                    # update the required files to no longer include calendar
                    req_lst = [
                        f for f in file_lst if f not in ["calendar", "calendar_dates"]
                    ]

        # check for the presence of all required files
        missing_lst = set(req_lst) - set(file_lst)

        # if there are missing files, then invalid
        if len(missing_lst) == len(req_lst):
            valid = False
            logging.error(
                f"Cannot locate any required files in the GTFS dataset {self.gtfs_folder.stem}."
            )

        elif len(missing_lst) > 0:
            valid = False
            logging.error(
                f"GTFS dataset, {self.gtfs_folder.stem}, is missing required files: {missing_lst}"
            )

        else:
            valid = True
            logging.debug(
                f"GTFS dataset, {self.gtfs_folder.stem}, includes all required files."
            )

        return valid

    @cached_property
    def crosstab_stop_trip(self) -> pd.DataFrame:
        """Data frame with crosstabs lookup between stops and trips."""
        df = self.stop_times.data[["stop_id", "trip_id"]].drop_duplicates()
        return df

    @cached_property
    def crosstab_trip_route(self) -> pd.DataFrame:
        """Data frame with crosstabs lookup between trips and routes. This is useful when attempting to associate trips
        to routes."""
        df = self.trips.data[["trip_id", "route_id"]].drop_duplicates()
        return df

    @cached_property
    def crosstab_trip_service(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between trips and services. This is useful when attempting to associate
        trips to calendar."""
        df = self.trips.data[["trip_id", "service_id"]].drop_duplicates()
        return df

    @cached_property
    def crosstab_route_agency(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between routes and agencies."""
        df = self.routes.data[["route_id", "agency_id"]].drop_duplicates()
        return df

    @cached_property
    def crosstab_stop_route(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between stops and routes."""
        df = (
            self.crosstab_stop_trip.join(
                self.crosstab_trip_route.set_index("trip_id"), on="trip_id", how="left"
            )[["stop_id", "route_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return df

    @cached_property
    def crosstab_stop_agency(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between trips and agency id."""
        df = (
            self.crosstab_stop_route.join(
                self.routes.data[["route_id", "agency_id"]].set_index("route_id"),
                on="route_id",
                how="left",
            )
            .drop(columns="route_id")
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return df

    @cached_property
    def crosstab_stop_service(self) -> pd.DataFrame:
        """Data frame with crosstab lookup between stops and service_id. This is useful when attempting to associate
        between stops and calendar."""
        df = self.crosstab_stop_trip.join(
            self.crosstab_trip_service.set_index("trip_id"), on="trip_id"
        )[["stop_id", "service_id"]].drop_duplicates()
        return df

    @cached_property
    def stops_with_calendar(self) -> pd.DataFrame:
        """
        Stops with service days from calendar added. Since multiple trips (and routes) service a single stop, if the
        stop has service by *any* trip on a day, the stop then has service.
        """
        df_svc = (
            self.crosstab_stop_service.join(
                self.calendar.data.set_index("service_id"), on="service_id", how="outer"
            )
            .drop(columns=["service_id", "start_date", "end_date"])
            .groupby("stop_id")
            .any()
            .astype("Int64")
        )

        df = self.stops.data.join(df_svc, on="stop_id", how="left")
        return df

    @cached_property
    def stops_with_agency(self) -> pd.DataFrame:
        """
        Stops data frame with agency id and name. Since multiple agencies can serve a single stop, stops may be
        listed more than once.
        """
        df = (
            self.stops.data.join(
                self.crosstab_stop_agency.set_index("stop_id"),
                on="stop_id",
                how="outer",
            )
            .join(self.agency.data.set_index("agency_id"), on="agency_id", how="left")
            .sort_values(["stop_id", "agency_name"])
            .reset_index(drop=True)
        )
        return df
