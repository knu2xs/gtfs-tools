import logging
import os
from pathlib import Path
from typing import Union

import arcpy

from arcgis.features import GeoAccessor, GeoSeriesAccessor
import pandas as pd

__all__ = ["add_dataframe_to_feature_class"]


def add_dataframe_to_feature_class(
    data_frame: pd.DataFrame, feature_class: Union[str, Path]
) -> Path:
    """
    Add features to an already existing feature class from a Pandas data frame.

    Args:
        data_frame: Pandas data frame comprised of features.
        feature_class: Path to the feature class to append records to.
    """
    # make sure path is string so geoprocessing works
    if isinstance(feature_class, Path):
        feature_class = str(feature_class)

    # get the geometry column and columns to use for insert cursor if spatially enabled data frame
    if data_frame.spatial.validate():
        geom_col = data_frame.spatial.name
        df_cols = [c for c in data_frame.columns if c != geom_col]
    else:
        geom_col = None
        df_cols = list(data_frame.columns)

    # get columns names for the feature class
    fc_cols = [f.name for f in arcpy.ListFields(feature_class)]

    # get a list of fields to use, those missing and those not being used
    icur_cols = [c for c in fc_cols if c in df_cols]
    missing_cols = [c for c in fc_cols if c not in df_cols and c.lower() != "shape"]
    unused_cols = [c for c in df_cols if c not in fc_cols]

    # report if any columns not in the source
    if len(missing_cols) > 0:
        logging.info(
            f"{len(missing_cols)} columns are in the target feature class, but not in the source data frame, and "
            f"will not be populated with values {missing_cols}"
        )

    # report if any columns are not being used from teh source
    if len(unused_cols) > 0:
        logging.info(
            f"{len(unused_cols)} columns are in the source data frame, but not in the target feature class, so the "
            f"data in these columns will to be used {unused_cols}"
        )

    # if not needing to worry about a geometry column, input schema and insert cursor schema is identical
    if geom_col is None:
        df = data_frame.loc[:, icur_cols]

    # if needing to consider schema, handle geometry accordingly for insert schema
    else:
        # get the data frame data organized for the insert cursor
        df = data_frame.loc[:, icur_cols + [geom_col]]

        # add geometry to insert cursor columns
        icur_cols = icur_cols + ["SHAPE@"]

        # convert the geometry objects to arcpy geometry objects
        df[geom_col] = df[geom_col].geom.as_arcpy

    # if being used in Pro, provide progress status
    arcpy.SetProgressor(
        "step",
        message=f"Inserting data into {os.path.basename(feature_class)}",
        min_range=0,
        max_range=df.shape[0],
        step_value=1,
    )

    # create a cursor to insert data
    with arcpy.da.InsertCursor(feature_class, icur_cols) as icur:
        # iterate the rows in the input data frame
        for row_tpl in df.itertuples():
            # get a list of the row values filling null with None
            row = [None if pd.isnull(val) else val for val in row_tpl[1:]]

            # insert the row into the feature class
            icur.insertRow(row)

            arcpy.SetProgressorPosition()

    # reset ArcGIS Pro progress bar
    arcpy.ResetProgressor()

    return Path(feature_class)
