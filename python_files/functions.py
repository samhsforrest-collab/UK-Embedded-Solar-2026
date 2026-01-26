# Environments
import requests
import pandas as pd
import pytz
from datetime import datetime
from pvlive_api import PVLive
import time
from datetime import datetime, timedelta, timezone
import requests_cache
from retry_requests import retry
import openmeteo_requests
from dates_and_gsp_selection import validate_date_range

import pytz
from datetime import datetime
import time
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta


# Initiating PVLive API as per GIT repo instructions: https://github.com/SheffieldSolar/PV_Live-API
pvl = PVLive(
    retries=3, # Optionally set the number of retries when intermittent issues are encountered
    proxies=None, # Optionally pass a dict of proxies to use when making requests
    ssl_verify=True, # Optionally disable SSL certificate verification (not advised!)
    domain_url="api.pvlive.uk", # Optionally switch between the prod and FOF APIs
)

def load_mwp(region="gsp", include_history=True):
    """
    Load and return the MWp deployment dataframe as mwp_df via pvl.deployment.
    """
    mwp_df = pvl.deployment(region=region, include_history=include_history)
    return mwp_df
    
mwp_df = load_mwp() # loading the capacity df

def load_gsp(gsp_path="python_files\data\gsp_info.csv"):
    """
    Load and return the GSP info dataframe as gsp_df from CSV,
    filtered to only GSPs known to PVLive (pvl.gsp_ids).
    """
    pvl = PVLive()
    valid_ids = set(pvl.gsp_ids)

    gsp_df = pd.read_csv(gsp_path)

    if 'gsp_id' in gsp_df.columns:
        # coerce non-numeric to NaN, drop those rows, cast to int, then filter by PVLive ids
        gsp_df['gsp_id_num'] = pd.to_numeric(gsp_df['gsp_id'], errors='coerce')
        gsp_df = gsp_df[gsp_df['gsp_id_num'].notna()].copy()
        gsp_df['gsp_id_num'] = gsp_df['gsp_id_num'].astype(int)
        gsp_df = gsp_df[gsp_df['gsp_id_num'].isin(valid_ids)].drop(columns=['gsp_id_num']).reset_index(drop=True)

    return gsp_df
    
gsp_df = load_gsp() # loading the gsp locations df

def merge_gsp_location(mwp_df, gsp_df, gsp_col_mwp='GSPs', gsp_col_gsp='gsp_name'):
    """
    Return a copy of mwp_df with columns gsp_lat, gsp_lon, region_name merged from gsp_df.
    Matching is done case-insensitive and with whitespace stripped.
    Remove the 'unkown' rows from the mwp_df - presumably misspelling of unknown.
    Drop any rows with missing values.
    
    """
    # Make copies to avoid mutating inputs
    mwp = mwp_df.copy()
    gsp = gsp_df.copy()

    # Normalize join keys by aligning to string, stripping and putting in upper case
    mwp['_gsp_key'] = mwp[gsp_col_mwp].astype(str).str.strip().str.upper()
    gsp['_gsp_key'] = gsp[gsp_col_gsp].astype(str).str.strip().str.upper()

    # Select only the columns we want to bring across (plus join key)
    to_merge = gsp[['_gsp_key', 'gsp_id', 'gsp_lat', 'gsp_lon', 'region_name', 'pes_id']].drop_duplicates('_gsp_key')    
    merged = mwp.merge(to_merge, on='_gsp_key', how='left') # Left merge so all mwp rows are kept
    merged = merged.dropna(how='any')   # drop all rows where ther are NaN values - return only the 299 intersection GSPs
    merged = merged[merged[gsp_col_mwp] != 'unkown']  # return the df where not equal to unkown
    merged = merged.drop(columns=['_gsp_key'])  # Drop linking key

    return merged

merged_df = merge_gsp_location(mwp_df, gsp_df) # merge capacity growth and locations

def gsp_locations(merged_df, gsp_col='GSPs'):
    """
    Return a DataFrame with one row per unique GSP containing
    gsp_col, gsp_lat, gsp_lon, region_name, pes_id, and a combined
    'GSP_region' column formatted "GSPs | region_name".
    """
    gsp_locations_list = (
        merged_df
        .drop_duplicates(subset=[gsp_col])[[gsp_col, 'gsp_id', 'gsp_lat', 'gsp_lon', 'region_name', 'pes_id']]
        .reset_index(drop=True)
    )
    gsp_locations_list['GSP_region'] = gsp_locations_list[gsp_col].astype(str) + ' | ' + gsp_locations_list['region_name'].astype(str)
    return gsp_locations_list

gsp_locations_list = gsp_locations(merged_df) # merge capacity and locations without capacity growth over time

def wide_cumul_capacity(merged_df, time_col='install_month', gsp_col='GSPs', value_col='cumul_capacity_mwp'):
    """
    Return a DataFrame with:
    - one column for the time (time_col) monthly intervals
    - one column per GSP (column name = GSP identifier)
    - cells = value_col (cumulative capacity MWP)
    """
    import pandas as pd
    df = merged_df.copy()
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception:
        pass
    wide = df.pivot_table(index=time_col, columns=gsp_col, values=value_col, aggfunc='first')
    wide = wide.reset_index()  # make time a regular column
    return wide

capacity_growth_all_gsps = wide_cumul_capacity(merged_df) # wide capacity growth df for all time and all gsps
