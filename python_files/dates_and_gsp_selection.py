# utils.py
from datetime import datetime, time
from dateutil.relativedelta import relativedelta
import pytz

def validate_date_range(start_date, end_date, min_months=1, tz=pytz.UTC):
    if isinstance(start_date, datetime): sd = start_date.date()  # Accept either datetime or date for start_date; normalize to date
    else: sd = start_date  # Accept either datetime or date for start_date; normalize to date
    if isinstance(end_date, datetime): ed = end_date.date()  # Accept either datetime or date for end_date; normalize to date
    else: ed = end_date  # Accept either datetime or date for end_date; normalize to date
    start_dt = datetime.combine(sd, time(0,0)).replace(tzinfo=tz)  # Combine date with specific time and set timezone: start at 00:00
    end_dt = datetime.combine(ed, time(23,30)).replace(tzinfo=tz)  # Combine date with specific time and set timezone: end at 23:30
    if end_dt < start_dt: raise ValueError("End date must not be before start date.")  # Validate that end is not before start
    if end_dt < (start_dt + relativedelta(months=min_months)): raise ValueError(f"End date must be at least {min_months} month(s) after start date.")  # Validate range spans at least min_months
    return start_dt, end_dt  # Return datetime bounds with timezone applied

def get_gsp_id_from_region(gsp_locations_list, gsp_region):
    """
    Given gsp_locations_list DataFrame and a GSP_region string, return gsp_id as int.
    Raises ValueError if not found or invalid input.
    """
    if gsp_region is None: raise ValueError("No GSP_region provided.")  # Ensure a region string was provided
    if 'GSP_region' not in gsp_locations_list.columns or 'gsp_id' not in gsp_locations_list.columns: raise ValueError("gsp_locations_list must contain 'GSP_region' and 'gsp_id' columns.")  # Ensure required columns exist
    row = gsp_locations_list[gsp_locations_list['GSP_region'] == gsp_region]  # Filter rows matching the requested region
    if row.empty: raise ValueError(f"GSP_region '{gsp_region}' not found.")  # If no match, raise an error
    gsp_id = row.iloc[0]['gsp_id']  # Extract the gsp_id from the first matching row
    try:
        return int(gsp_id)  # Attempt to convert to int
    except Exception:
        raise ValueError(f"gsp_id for '{gsp_region}' is not convertible to int: {gsp_id}")  # Raise on conversion failure
