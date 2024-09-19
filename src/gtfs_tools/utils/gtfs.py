import datetime
import logging
from pathlib import Path

import pandas as pd


def interpolate_stop_times(stop_times_df: pd.DataFrame) -> pd.DataFrame:
    """Helper to preprocess stop times arrival and departure columns before validation handling over 24 hours caveat."""

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
        calendar_dates: Data frame created from calendar_dates.txt.
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


def get_route_types() -> pd.DataFrame:
    """Get a route types dataframe with translations between european codes and standard GTFS route type codes."""
    # create a data frame of descriptions for the route types
    _route_type_pth = (
        Path(__file__).parent.parent / "assets" / "gtfs_modality_translation.csv"
    )

    route_type_df = pd.read_csv(
        filepath_or_buffer=_route_type_pth,
        names=["route_type", "route_type_desc", "route_type_carto"],
        dtype={"route_type": str, "route_type_desc": str, "route_type_carto": str},
        index_col="route_type",
    )

    return route_type_df
