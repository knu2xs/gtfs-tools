"""
This is a stubbed out test file designed to be used with PyTest, but can 
easily be modified to support any testing framework.
"""
import itertools
from pathlib import Path

import pandas as pd
import sys

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

# all the gtfs datasets to test
gtfs_parent = dir_raw / "gtfs_olympia"
gtfs_dir_lst = [p for p in gtfs_parent.glob("*") if p.is_dir()]


class TestGtfsDatasets:
    @pytest.mark.parametrize("gtfs_dir", gtfs_dir_lst)
    def test_gtfs_instantiation(self, gtfs_dir):
        gtfs = gtfs_tools.gtfs.GtfsDataset(gtfs_dir)
        assert isinstance(gtfs, gtfs_tools.gtfs.GtfsDataset)

    @pytest.mark.parametrize(
        "gtfs_directory,file_property",
        itertools.product(gtfs_dir_lst, file_properties),
    )
    def test_file_data_df(self, gtfs_directory, file_property):
        gtfs_pth = Path(gtfs_directory)
        gtfs = gtfs_tools.gtfs.GtfsDataset(gtfs_pth)
        gtfs_prop = getattr(gtfs, file_property)
        assert isinstance(gtfs_prop, gtfs_tools.gtfs.GtfsFile)

        df = gtfs_prop.data
        assert isinstance(df, pd.DataFrame)
