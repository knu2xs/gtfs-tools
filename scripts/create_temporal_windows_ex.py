from pathlib import Path
from gtfs_tools.gtfs import GtfsDataset

gtfs_pth = Path(
    r"D:\projects\GTFS-Publishing\data\interim\utah_transit_authority_gtfs_2zip_2024-06-18T164407\gtfs"
)
gtfs = GtfsDataset(gtfs_pth)

tmprl_win_df = gtfs.trips.temporal_windows

assert tmprl_win_df.any().all()
