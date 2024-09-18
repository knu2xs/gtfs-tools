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
        if gtfs_prop in ["shapes", "stops"]:
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
