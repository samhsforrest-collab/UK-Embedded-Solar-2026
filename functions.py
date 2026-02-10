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
import streamlit as st
from io import BytesIO
from typing import Optional
from pvlive_api import PVLive

# Initiating PVLive API as per GIT repo instructions: https://github.com/SheffieldSolar/PV_Live-API
pvl = PVLive(
    retries=3, # Optionally set the number of retries when intermittent issues are encountered
    proxies=None, # Optionally pass a dict of proxies to use when making requests
    ssl_verify=True, # Optionally disable SSL certificate verification (not advised!)
    domain_url="api.pvlive.uk", # Optionally switch between the prod and FOF APIs
)

def load_mwp(pvl: Optional[PVLive] = None,
               region: str = "gsp",
               include_history: bool = True,
               by_system_size: bool = False) -> pd.DataFrame:
    """
    Return MWp deployment DataFrame by scanning available PVLive filenames and
    downloading the best match (more robust than PVLive.deployment when naming changed).

    Parameters
    - pvl: optional PVLive instance. If None, a new PVLive() will be created.
    - region: "gsp" or "llsoa".
    - include_history: include monthly history if True.
    - by_system_size: include system-size breakdown (only valid for region="gsp").

    Returns:
    - pd.DataFrame (empty if no match or on error).
    """
    if pvl is None:
        pvl = PVLive()

    try:
        deployment_datasets, releases = pvl._get_deployment_releases()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch deployment releases: {e}")

    # Build tokens to look for in filenames
    region_token = f"{pvl.gsp_boundaries_version}_GSP" if region == "gsp" else region
    # Fallback: sometimes the filenames embed a different boundaries datestamp (e.g. '20251204_GSP')
    history_token = "_and_month" if include_history else ""
    system_size_token = "_and_system_size" if by_system_size else ""
    target_fragment = f"_capacity_by_{region_token}{history_token}{system_size_token}"

    chosen_release = None
    chosen_filename = None

    for rel in releases:
        filenames = list(deployment_datasets.get(rel, {}).keys())
        # first try flexible containment match using constructed fragment
        matches = [f for f in filenames if target_fragment in f]
        # fallback: look for capacity_by entries that contain the region token or region string
        if not matches:
            matches = [f for f in filenames if ("_capacity_by_" in f) and (region_token in f or f"_capacity_by_{region}" in f)]
        # final fallback: any capacity_by for the region
        if not matches:
            matches = [f for f in filenames if ("_capacity_by_" in f) and (region in f)]
        if matches:
            chosen_release = rel
            chosen_filename = matches[0]
            break

    if not chosen_filename:
        # no match found
        return pd.DataFrame()

    # Download and parse chosen file
    try:
        url = deployment_datasets[chosen_release][chosen_filename]
        response = pvl._fetch_url(url, parse_json=False)
        mock_file = BytesIO(response.content)
        kwargs = dict(parse_dates=["install_month"]) if include_history else {}
        df = pd.read_csv(mock_file, compression={"method": "gzip"}, **kwargs)
        df.insert(0, "release", chosen_release)
        if "dc_capacity_MWp" in df.columns:
            df.rename(columns={"dc_capacity_MWp": "dc_capacity_mwp"}, inplace=True)
        if "system_count" in df.columns:
            df.system_count = df.system_count.astype("Int64")
        df.dropna(how="any", inplace=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to download/parse PVLive file '{chosen_filename}': {e}")


mwp_df = load_mwp()

def load_gsp(gsp_path="data/gsp_info.csv"):
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
