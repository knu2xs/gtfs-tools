"""
This is a stubbed out test file designed to be used with PyTest, but can 
easily be modified to support any testing framework.
"""
import itertools
import logging
from pathlib import Path
import sys

import pandas as pd
import pytest

# get paths to useful resources - notably where the src directory is
self_pth = Path(__file__)
dir_test = self_pth.parent
dir_prj = dir_test.parent
dir_src = dir_prj / "src"

dir_data = dir_prj / "data"
dir_raw = dir_data / "raw"

# insert the src directory into the path and import the project package
sys.path.insert(0, str(dir_src))
import gtfs_tools

# all the files to look for
file_properties = [
    "agency",
    "calendar",
    "calendar_dates",
    "frequencies",
    "routes",
    "shapes",
    "stops",
    "stop_times",
    "trips",
]


def test_gtfs_init():
    pth = dir_raw / "gtfs_olympia" / "mdb_source_id=268"
    gtfs = gtfs_tools.gtfs.GtfsDataset(pth, standardize_route_types=True)
    assert isinstance(gtfs, gtfs_tools.gtfs.GtfsDataset)


def test_stops_data():
    pth = dir_raw / "gtfs_olympia" / "mdb_source_id=268"
    gtfs = gtfs_tools.gtfs.GtfsDataset(pth, standardize_route_types=True)
    stops = gtfs.stops
    assert isinstance(stops, gtfs_tools.gtfs.GtfsStops)
    stops_df = stops.data
    assert isinstance(stops_df, pd.DataFrame)
    stops_sedf = stops.sedf
    assert stops_sedf.spatial.validate()


def test_swiss_modality():
    pth = r"\\DevBA00007\data\gtfs_publishing\raw\esri_switzerland_gtfsfp202520240923zip_2024-09-26_11_58\gtfs"
    gtfs = gtfs_tools.gtfs.GtfsDataset(pth)
    modality_df = gtfs.stops.modalities
    assert isinstance(modality_df, pd.DataFrame)


class TestGtfsNL:
    @pytest.fixture(scope="class")
    def gtfs_source(self):
        gtfs_pth = Path(
            r"\\devba00007\data\gtfs_publishing\interim_quarantine\Esri_NL_gtfsnlzip_2024-08-13_16_01\gtfs"
        )
        gtfs = gtfs_tools.gtfs.GtfsDataset(gtfs_pth)
        return gtfs

    def test_gtfs_stop_modalities(self, gtfs_source):
        assert isinstance(gtfs_source.stops, gtfs_tools.gtfs.GtfsStops)
        modalities_df = gtfs_source.stops.modalities
        assert modalities_df[modalities_df["route_type"].isnull()].shape[0] <= 45

    def test_gtfs_stop_agency(self, gtfs_source):
        assert isinstance(gtfs_source.stops, gtfs_tools.gtfs.GtfsStops)
        agency_df = gtfs_source.stops.agency
        assert agency_df[agency_df["agency_name"].isnull()].shape[0] == 0

    def test_gtfs_route_read_agency_id(self, gtfs_source):
        route_df = gtfs_source.routes.data
        assert ~route_df["agency_id"].isnull().any()


class TestGtfsAddLookupColumns:
    @pytest.fixture(scope="class")
    def gtfs_source(self):
        gtfs_pth = Path(
            r"\\DevBA00007\data\gtfs_publishing\interim\Grand_County_Colorado_Bus_winterparkcousgtfszip_2024-06-19_10_43_48\gtfs"
        )
        gtfs = gtfs_tools.gtfs.GtfsDataset(gtfs_pth)
        return gtfs

    def test_add_agency_route(self, gtfs_source):
        routes_df = routes_df = gtfs_tools.utils.gtfs.add_agency_name_column(
            gtfs_source.routes.data, gtfs_source.agency.data
        )
        assert routes_df["agency_name"].notnull().all()

    def test_add_modality_std_routes(self, gtfs_source):
        routes_df = gtfs_tools.utils.gtfs.add_standardized_modality_column(
            gtfs_source.routes.data
        )
        assert routes_df["route_type_std"].notnull().all()


class TestGtfsDatasets:
    # all the gtfs datasets to test
    gtfs_parent = dir_raw / "gtfs_msp_cbsa"
    gtfs_dir_lst = [p for p in gtfs_parent.glob("*") if p.is_dir()]

    @pytest.mark.parametrize("gtfs_dir", gtfs_dir_lst)
    def test_gtfs_instantiation(self, gtfs_dir):
        gtfs = gtfs_tools.gtfs.GtfsDataset(gtfs_dir)
        assert isinstance(gtfs, gtfs_tools.gtfs.GtfsDataset)

    @pytest.mark.parametrize(
        "gtfs_directory,file_property",
        itertools.product(gtfs_dir_lst, file_properties),
    )
    def test_file_data_df(self, gtfs_directory, file_property):
        # ensure path
        gtfs_pth = Path(gtfs_directory)

        # get a GTFS object instance
        gtfs = gtfs_tools.gtfs.GtfsDataset(gtfs_pth)

        # get the property for the test
        gtfs_prop = getattr(gtfs, file_property)

        # ensure the property is an instance of the parent class - ensures was instantiated correctly
        assert isinstance(gtfs_prop, gtfs_tools.gtfs.GtfsFile)

        # create the data frame
        df = gtfs_prop.data
        assert isinstance(df, pd.DataFrame)

        # for the two properties with spatial, ensure spatially enabled data frame is available and valid
        if file_property in ["shapes", "stops", "trips", "routes"]:
            sedf = getattr(gtfs_prop, "sedf")
            assert sedf.spatial.validate()

        # since inferring all the intermediate values, all stop times should be populated
        if file_property == "stop_times":
            assert ~df["arrival_time"].isnull().all()
            assert ~df["departure_time"].isnull().all()

        # ensure both columns are in the agency file and populated
        if file_property == "agency":
            assert (
                len(
                    df[
                        (~df["agency_id"].isnull()) & (~df["agency_name"].isnull())
                    ].index
                )
                >= 1
            )
