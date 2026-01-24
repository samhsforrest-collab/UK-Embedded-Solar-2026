# utils.py
from datetime import datetime, time
from dateutil.relativedelta import relativedelta
import pytz

def validate_date_range(start_date, end_date, min_months=1, tz=pytz.UTC):
    if isinstance(start_date, datetime):
        sd = start_date.date()
    else:
        sd = start_date
    if isinstance(end_date, datetime):
        ed = end_date.date()
    else:
        ed = end_date
    start_dt = datetime.combine(sd, time(0,0)).replace(tzinfo=tz)
    end_dt = datetime.combine(ed, time(23,30)).replace(tzinfo=tz)
    if end_dt < start_dt:
        raise ValueError("End date must not be before start date.")
    if end_dt < (start_dt + relativedelta(months=min_months)):
        raise ValueError(f"End date must be at least {min_months} month(s) after start date.")
    return start_dt, end_dt

def get_gsp_id_from_region(gsp_locations_list, gsp_region):
    """
    Given gsp_locations_list DataFrame and a GSP_region string, return gsp_id as int.
    Raises ValueError if not found or invalid input.
    """
    if gsp_region is None:
        raise ValueError("No GSP_region provided.")
    # ensure column exists
    if 'GSP_region' not in gsp_locations_list.columns or 'gsp_id' not in gsp_locations_list.columns:
        raise ValueError("gsp_locations_list must contain 'GSP_region' and 'gsp_id' columns.")

    row = gsp_locations_list[gsp_locations_list['GSP_region'] == gsp_region]
    if row.empty:
        raise ValueError(f"GSP_region '{gsp_region}' not found.")
    gsp_id = row.iloc[0]['gsp_id']
    try:
        return int(gsp_id)
    except Exception:
        raise ValueError(f"gsp_id for '{gsp_region}' is not convertible to int: {gsp_id}")
