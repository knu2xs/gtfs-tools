import logging

import pandas as pd


def interpolate_stop_times(stop_times_df: pd.DataFrame) -> pd.DataFrame:
    """Helper to preprocess stop times arrival and departure columns before validation handling over 24 hours caveat."""

    # flag for providing interpolation messaging
    stop_times_cnt = len(stop_times_df.index)

    # apply interpolation to both arrival and departure columns
    for col in ["arrival_time", "departure_time"]:
        if col in stop_times_df.columns:
            # get the count by trip to ascertain if valid, has more than two times
            df_col = stop_times_df[["trip_id", col]]

            # get the count of valid (non-null) values for each trip
            df_cnt = (
                stop_times_df[["trip_id", col, "stop_id"]].groupby("trip_id").count()
            )
            df_cnt.columns = ["valid_count", "count"]

            # add the invalid count onto the valid count data frame
            df_valid = df_col.join(df_cnt, on="trip_id", how="left")
            df_valid["invalid_count"] = df_valid["count"] - df_valid["valid_count"]

            # create data frame flags for valid and invalid trips
            df_valid["trip_valid"] = df_valid["valid_count"] >= 2
            df_valid["trip_invalid"] = df_valid["valid_count"] < 2

            # get the valid, invalid and update trip counts
            stop_valid_cnt = len(df_valid[df_valid["trip_valid"]].index)
            stop_invalid_cnt = len(df_valid[df_valid["trip_invalid"]].index)
            stop_update_cnt = len(
                df_valid[(df_valid["trip_valid"]) & (df_valid[col].isnull())].index
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
