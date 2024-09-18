import datetime
import math


def hh_mm_to_timedelta(timestamp_str: str) -> datetime.timedelta:
    """Convert an HH:MM timestamp to a timedelta object."""
    # swap in a zero length string if not the right data type, so it gets handled correctly
    if not isinstance(timestamp_str, str):
        timestamp_str = ""

    # peel off hours for adjustment
    split_parts = timestamp_str.split(":")

    # ensure parts are all present
    if not len(split_parts) == 3:
        ret_val = None

    else:
        # get the hours, minutes and seconds as integers
        hh, mm, ss = [int(pt) for pt in split_parts]

        # get the days
        days = int(math.floor(hh / 24))

        # get the leftover hours
        hr_mod = int(hh % 24)

        # create a timedelta object
        ret_val = datetime.timedelta(days=days, hours=hr_mod, minutes=mm, seconds=ss)

    return ret_val
