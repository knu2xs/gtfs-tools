# GTFS-Tools Documentation

GTFS-Tools arose out of a need for loading a *lot* of GTFS static datasets with the ability to flexibly add information
from the various tables included in a GTFS dataset. Since there is a *lot* of flexibility in the GTFS specification,
being able to get the information needed from a large variety of datasets requires flexibly handing the huge
variation in implementations. Hence, this Python package to make processing these datasets much easier.

## Example

GTFS-Tools enables quickly interrogating and navigating the data from a GTFS dataset.

```python
from pathlib import Path

from gtfs_tools.gtfs import GtfsDataset

# path to directory with files comprising GTFS dataset or GTFS archve
gtfs_pth = r'C:\data\transit_data\raw\valley_metro\gtfs.zip'

# instantiate the GTFS Dataset object
gtfs = GtfsDataset(gtfs_pth)

# get the stops as a Spatially Enabled Dataframe
stops_sedf = gtfs.stops.sedf

# get the routes as a Spatially Enabled Dataframe - this creates geometry from shapes -> trips -> routes
routes_sedf = gtfs.routes.sedf
```

The real power of GTFS-Tools is in the ease of combining information from different tables in the GTFS
dataset. For instance, if wanting to include information from the agency table on stops and routes, 
GTFS-Tools provides a lookup dataframe for doing this. This lookup dataframe provides the crosstab
lookup needed to merge data between the tables.

```python
# get stops with agency infomation using relationship dataframe
stops_sedf = (
    gtfs.stops.sedf
    .merge(gtfs.lookup_stop_agency, on='stop_id', how='left')
    .merge(gtfs.agency.data, on='agency_id', how='left')
)

# set the geometry after the merge so the dataframe is still a Spatially Enabled Dataframe
stops_sedf = stops_sedf.spatial.set_geometry('SHAPE')

# get stops with agency infomation using relationship dataframe
routes_sedf = (
    gtfs.routes.sedf
    .merge(gtfs.lookup_route_agency, on='route_id', how='left')
    .merge(gtfs.agency.data, on='agency_id', how='left')
)

# set the geometry after the merge so the dataframe is still a Spatially Enabled Dataframe
routes_sedf = routes_sedf.spatial.set_geometry('SHAPE')
```