# Environments START
#=============================================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import base64

# import PVLive API requirements
from pvlive_api import PVLive

# import openmeteo API requirements
import requests_cache
from retry_requests import retry
import openmeteo_requests

import pytz
from datetime import datetime
import time
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

# ML environments
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#=============================================================================================
# Environments END

# Extra Functions START
#=============================================================================================
# Pull date and GSP selector
from dates_and_gsp_selection import validate_date_range, get_gsp_id_from_region

# Pull functions from functions
from functions import (
    load_mwp,
    load_gsp,
    merge_gsp_location,
    gsp_locations,
    wide_cumul_capacity,
)
# Call functions and create variables holidng dataframes for use in the app
mwp_df = load_mwp() # loading the capacity df
gsp_df = load_gsp() # loading the gsp locations df
merged_df = merge_gsp_location(mwp_df, gsp_df) # merge capacity growth and locations
gsp_locations_list = gsp_locations(merged_df) # merge capacity and locations without capacity growth over time
capacity_growth_all_gsps = wide_cumul_capacity(merged_df) # wide capacity growth df for all time and all gsps
#=============================================================================================
# Extra Functions END

# Page Configurations START
#=============================================================================================

# image banner and logo
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

banner_b64 = get_base64_image("assets/solar-image2.jpg")
st.image("assets/solar-image2.jpg", use_container_width =True)
st.sidebar.image("assets/logo.png", use_container_width=True)

st.set_page_config(page_title="Embedded Solar", page_icon="☀️", layout="wide")

#=============================================================================================
# Page Configurations END

# Sidebar Configurations START
#=============================================================================================
# sidebar for the selected GSP to evaluate

options = gsp_locations_list['GSP_region'].dropna().unique().tolist()
selected_gsp = st.sidebar.selectbox(
    "Select: GSP ID | Region",
    options,
    index=0,
    key="gsp_region_select",
    help="Choose a GSP region from the list"
)
# handle placeholder
if selected_gsp == "Click for options":
    st.sidebar.write("No GSP chosen yet")
    selected_row = pd.DataFrame()  # or None
else:
    selected_row = gsp_locations_list[gsp_locations_list['GSP_region'] == selected_gsp].reset_index(drop=True)
    st.sidebar.write("Selected:", selected_gsp)

selected_row = gsp_locations_list[gsp_locations_list['GSP_region'] == selected_gsp].reset_index(drop=True)

# Create a sidebar selector for the dates to choose from
# Create the minimum start date
min_start_date = datetime(2023, 1, 1).date()  # Converts it to a date object
min_end_date = datetime(2023, 2, 1).date()  # Converts it to a date object

start_date = st.sidebar.date_input(
    "Start Date",  # Label for the date input
    value=datetime.today(),  # Default value set to today's date
    min_value=min_start_date,  # Minimum allowed date
    max_value=datetime.today().date(),  # Maximum allowed date
    format="YYYY/MM/DD",  # Date format
    disabled=False,  # Whether the input should be disabled
    label_visibility="visible",  # Visibility of the label
    width="stretch"  # Width of the input element
)
end_date = st.sidebar.date_input(
    "End Date",  # Label for the date input
    value=datetime.today(),  # Default value set to today's date
    min_value=min_end_date,  # Minimum allowed date
    max_value=datetime.today().date(),  # Maximum allowed date
    format="YYYY/MM/DD",  # Date format
    disabled=False,  # Whether the input should be disabled
    label_visibility="visible",  # Visibility of the label
    width="stretch"  # Width of the input element
)

#=============================================================================================
# Sidebar Configurations END

# CREATE MAIN DATAFRAMES - FEATURE ENGINEERING - START
#=============================================================================================
# create function which makes start and end variables using date range from sidebar selector
try:
    start, end = validate_date_range(start_date, end_date)
except ValueError as e:
    st.error(str(e))
    st.stop()
# find gsp_id using get_gsp_id_from_region function for given sidebar GSP_region name selection 
gsp_id = get_gsp_id_from_region(gsp_locations_list, selected_gsp) # selects chosen GSP for use in the visuals

# Initiating PVLive API as per GIT repo instructions: https://github.com/SheffieldSolar/PV_Live-API
pvl = PVLive(
    retries=3, # Optionally set the number of retries when intermittent issues are encountered
    proxies=None, # Optionally pass a dict of proxies to use when making requests
    ssl_verify=True, # Optionally disable SSL certificate verification (not advised!)
    domain_url="api.pvlive.uk", # Optionally switch between the prod and FOF APIs
)

def download_generation_for_single_gsp(start, end, gsp_id, gsp_locations_list, include_national=False, extra_fields=""):
    """
    Return a DataFrame with:
    - generation data for selected period for one GSP
    - one column for the time (time_col) HH intervals
    - other columns for GSP identifiers
    """
    valid_ids = gsp_locations_list['gsp_id'].dropna().astype(int).unique()  # Get the valid gsp_ids from gsp_locations_list
    if gsp_id not in valid_ids:  # Check if the provided gsp_id is valid
        return f"Please select a GSP ID which appears in the GSP capacity list."

    # Fetch data for the specific GSP ID using between function from PVLive API Class
    generation_df = pvl.between(
        start=start,
        end=end,
        entity_type="gsp",
        entity_id=int(gsp_id),
        dataframe=True,
        extra_fields=extra_fields
    )

    # Interpolate up to 24 consecutive NaN values in the generation output - 12 in either direction  
    if generation_df is not None and not generation_df.empty:
        generation_df['datetime_gmt'] = pd.to_datetime(generation_df['datetime_gmt'])
        generation_df = generation_df.sort_values(['gsp_id', 'datetime_gmt']).set_index('datetime_gmt')
        generation_df['generation_mw'] = (
            generation_df.groupby('gsp_id')['generation_mw']
            .apply(lambda s: s.interpolate(method='time', limit=12, limit_direction='both'))
            .reset_index(level=0, drop=True)
        )
        generation_df = generation_df.reset_index()
        
    # Merge additional columns from gsp_locations_list
    gsp_info = gsp_locations_list[gsp_locations_list['gsp_id'] == gsp_id]
    if not gsp_info.empty:
        # Merge on gsp_id to include other columns like gsp_lat, gsp_lon, etc.
        generation_df = generation_df.merge(gsp_info, on='gsp_id', how='left')
        
    return generation_df

generation_df = download_generation_for_single_gsp(start, end, gsp_id, gsp_locations_list) # generation df for selected gsp

def get_capacity_data_single_gsp(gsp_id, merged_df):
    """
    Return a DataFrame containing capacity data for the specified GSP ID,
    along with month and year columns based on the install_month
    
    """
    # Filter the DataFrame for the specified GSP ID and add time-series columns for mathing with generation df
    capacity_data = merged_df[merged_df['gsp_id'] == gsp_id].copy()
    capacity_data = capacity_data[['install_month', 'cumul_capacity_mwp', 'GSPs', 'gsp_lat', 'gsp_lon', 'region_name', 'pes_id']]   # Keep relevant columns
    capacity_data['install_month'] = pd.to_datetime(capacity_data['install_month'])    # Convert install_month to datetime
    capacity_data['month'] = capacity_data['install_month'].dt.month     # Create 'month' and 'year' columns
    capacity_data['year'] = capacity_data['install_month'].dt.year
    capacity_data['day'] = capacity_data['install_month'].dt.day
    capacity_data['hour'] = capacity_data['install_month'].dt.hour
    return capacity_data.reset_index(drop=True)

capacity_data_single_gsp = get_capacity_data_single_gsp(gsp_id, merged_df) # add month and year to capacity single gsp

def add_capacity_to_generation(generation_df, capacity_data, tz='UTC', gen_fill_method='interpolate'):
    """
    Merge cumulative capacity into generation_df, then reindex to a full 30-minute UTC grid
    and interpolate missing generation_mw and capacity_mwp values. Returns DataFrame indexed by datetime_gmt (tz-aware).
    
    Parameters:
    - generation_df: DataFrame with 'datetime_gmt' and 'generation_mw'
    - capacity_data: DataFrame with ['month', 'year', 'cumul_capacity_mwp']
    - tz: timezone for indexing (default 'UTC')
    - gen_fill_method: 'interpolate' (default) or 'zero' to fill generation gaps
    """
    # Ensure datetime and timezone
    generation_df = generation_df.copy()
    generation_df['datetime_gmt'] = pd.to_datetime(generation_df['datetime_gmt'])
    if generation_df['datetime_gmt'].dt.tz is None:
        generation_df['datetime_gmt'] = generation_df['datetime_gmt'].dt.tz_localize(tz)
    else:
        generation_df['datetime_gmt'] = generation_df['datetime_gmt'].dt.tz_convert(tz)

    # Add time components
    generation_df['month'] = generation_df['datetime_gmt'].dt.month
    generation_df['year'] = generation_df['datetime_gmt'].dt.year
    generation_df['day'] = generation_df['datetime_gmt'].dt.day
    generation_df['hour'] = generation_df['datetime_gmt'].dt.hour

    # Merge capacity (by month/year) and forward-fill capacity
    merged_df = generation_df.merge(
        capacity_data[['month', 'year', 'cumul_capacity_mwp']],
        on=['month', 'year'],
        how='left'
    ).rename(columns={'cumul_capacity_mwp': 'capacity_mwp'})
    merged_df['capacity_mwp'] = merged_df['capacity_mwp'].ffill()

    # Set datetime index and select relevant columns, including ffill on specific fields
    merged_df = merged_df.set_index('datetime_gmt').sort_index()

    # Build full 30-minute index from min to max
    full_idx = pd.date_range(merged_df.index.min(), merged_df.index.max(), freq='30min', tz=tz)

    # Reindex to full grid
    merged_df = merged_df.reindex(full_idx)

    # Fill remaining NaNs for generation_mw
    if 'generation_mw' in merged_df.columns:
        if gen_fill_method == 'interpolate':
            merged_df['generation_mw'] = merged_df['generation_mw'].interpolate(limit_direction='both')
        else:
            merged_df['generation_mw'] = merged_df['generation_mw'].fillna(0)
    
    # Fill remaining NaNs for capacity_mw
    if 'capacity_mwp' in merged_df.columns:
        if gen_fill_method == 'interpolate':
            merged_df['capacity_mwp'] = merged_df['capacity_mwp'].interpolate(limit_direction='both')
        else:
            merged_df['capacity_mwp'] = merged_df['capacity_mwp'].fillna(0)
        
    # Forward fill for newly made specific non-numeric columns
    ffill_columns = ['gsp_id', 'GSPs', 'gsp_lat', 'gsp_lon', 'region_name', 'pes_id', 'GSP_region']
    for col in ffill_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].ffill()

    # Recompute time component columns from index for any newly created rows
    merged_df['month'] = merged_df.index.month
    merged_df['year'] = merged_df.index.year
    merged_df['day'] = merged_df.index.day
    merged_df['hour'] = merged_df.index.hour

    # Reset index name to 'datetime_gmt'
    merged_df.index.name = 'datetime_gmt'

    return merged_df.reset_index()

generation_and_capacity_single_gsp = add_capacity_to_generation(generation_df, capacity_data_single_gsp) # merged capacity and generation same time-series single gsp

def fetch_weather_for_generation_df(gen_df,
                                    cache_path=".cache",
                                    cache_expire=3600,
                                    retries=5,
                                    backoff_factor=0.2):
    """
    Extract lat/lon and date range from generation_and_capacity_single_gsp-like
    DataFrame (expects 'gsp_lat', 'gsp_lon', 'datetime_gmt'), call Open-Meteo via
    openmeteo_requests and return hourly weather DataFrame indexed by UTC datetime
    with added columns: year, month, day, hour.
    """
    # Extract lat/lon from first non-null row
    row = gen_df[['gsp_lat', 'gsp_lon']].dropna().iloc[0]
    latitude, longitude = float(row['gsp_lat']), float(row['gsp_lon'])

    # Derive start/end dates (YYYY-MM-DD)
    start_date = gen_df['datetime_gmt'].min().normalize().strftime('%Y-%m-%d')
    end_date = gen_df['datetime_gmt'].max().normalize().strftime('%Y-%m-%d')

    # Setup client with cache + retry
    cache_session = requests_cache.CachedSession(cache_path, expire_after=cache_expire)
    retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
    client = openmeteo_requests.Client(session=retry_session)
    if not hasattr(client, "_close_session"):
        client._close_session = False  # avoid destructor AttributeError

    hourly_vars = [
        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
        "direct_normal_irradiance", "global_tilted_irradiance",
        "shortwave_radiation_instant", "direct_radiation_instant", "diffuse_radiation_instant",
        "direct_normal_irradiance_instant", "global_tilted_irradiance_instant",
    ]

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
    }

    responses = client.weather_api(url, params=params)
    if not responses:
        return pd.DataFrame()

    response = responses[0]
    hourly = response.Hourly()
    hourly_arrays = [hourly.Variables(i).ValuesAsNumpy() for i in range(len(hourly_vars))]

    start_ts = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    end_ts = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    interval_seconds = int(hourly.Interval())
    dates = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq=pd.Timedelta(seconds=interval_seconds),
        inclusive="left"
    )

    data = {"date": dates}
    for name, arr in zip(hourly_vars, hourly_arrays):
        data[name] = arr[: len(dates)]

    df = pd.DataFrame(data).set_index("date")

    # Add year, month, day, hour columns from the index
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour

    return df

weather_df = fetch_weather_for_generation_df(generation_and_capacity_single_gsp) # weather df for the selected gsp and time period

def merge_generation_with_weather(generation_df, weather_df):
    """
    Merge generation_and_capacity_single_gsp dataframe with weather_df.
    Only brings generation_mw and capacity_mwp from generation_df.
    - generation_df: must contain ['year','month','day','hour','generation_mw','capacity_mwp']
      (two half-hour rows per hour; this function averages per hour).
    - weather_df: indexed by UTC datetime and must contain columns ['year','month','day','hour']
    Returns: DataFrame with weather_df index and merged averaged columns:
             generation_mw, capacity_mwp (aligned to weather hourly rows).
    """
    # Select only required columns
    cols = ['year', 'month', 'day', 'hour', 'generation_mw', 'capacity_mwp']
    gen = generation_and_capacity_single_gsp[cols].copy()
    grouped = gen.groupby(['year', 'month', 'day', 'hour'], as_index=False)[['generation_mw', 'capacity_mwp']].mean() # Aggregate to hourly by mean
    weather = weather_df.copy().reset_index()  # bring datetime index to column named 'date'
    merged = weather.merge(grouped, on=['year', 'month', 'day', 'hour'], how='left') # Merge with weather_df on year,month,day,hour
    
    # Failsafe itnterpolate remaining NaN values in generation_mw and capacity_mwp
    if 'generation_mw' in merged.columns:
        merged['generation_mw'] = merged['generation_mw'].interpolate(limit_direction='both')

    if 'capacity_mwp' in merged.columns:
        merged['capacity_mwp'] = merged['capacity_mwp'].interpolate(limit_direction='both')
   
    # restore datetime index
    merged = merged.set_index('date')

    return merged
    
gen_weather_merged_df = merge_generation_with_weather(generation_and_capacity_single_gsp, weather_df)

# function for extracting the winning XGBoost model 
def prepare_and_fit_xgboost(df, target_col='generation_mw', test_size=0.2, random_state=101):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([target_col], axis=1),
        df[target_col],
        test_size=test_size,
        random_state=random_state
    )

    # Define the XGBoost pipeline
    pipeline = Pipeline([
        ("feat_selection", SelectFromModel(XGBRegressor(random_state=42, n_estimators=50, n_jobs=-1, verbosity=0))),
        ("model", XGBRegressor(
            random_state=42,
            n_estimators=50,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            n_jobs=-1,
            verbosity=0
        )),
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_train, X_test, y_train, y_test

# Create variables from the XGBoost model
pipeline, X_train, X_test, y_train, y_test = prepare_and_fit_xgboost(gen_weather_merged_df)

#=============================================================================================
# CREATE MAIN DATAFRAMES - FEATURE ENGINEERING - END

# Create Tabs START
#=============================================================================================
tab1, tab2 = st.tabs(["Selected GSP Stats", "UK Embedded Solar Capacity"])
#=============================================================================================
# Create Tabs END

# Tab 1 START
#=============================================================================================
with tab1:

    # Show Results of Selected GSP
    gsp_region = gsp_locations_list[gsp_locations_list['gsp_id'].astype(int) == int(gsp_id)].iloc[0]['GSP_region']
    st.subheader(f"{gsp_region} GSP: {start_date} - {end_date}")
    st.markdown("---")
    last_row = capacity_growth_all_gsps.iloc[-1]  # Last row values

    # MAP OF CAPACITY INSTALLED START
    # Create a Capacity DataFrame where each GSP gets the last capacity values.
    latest_capacity_dict = {
        'GSPs': gsp_locations_list['GSPs'],  # GSP identifiers
        'Latest Installed Capacity (MW)': [last_row[gsp] for gsp in gsp_locations_list['GSPs']]  # GSPs must match column names
    }
    latest_capacity_df = pd.DataFrame(latest_capacity_dict)
    gsp_locations_with_capacity = gsp_locations_list.merge(latest_capacity_df, on='GSPs', how='left')

    # Create a scatter map with sizes based on Latest Installed Capacity
    fig = px.scatter_map(
        gsp_locations_with_capacity,
        lat="gsp_lat",
        lon="gsp_lon",
        hover_name="region_name",  # Show region name on hover
        size="Latest Installed Capacity (MW)",  # Scale point size based on this capacity
        size_max=20,  # Max size for points
        title="GSP Locations with Latest Installed Capacity",
        map_style="open-street-map"
    )

    # Show the figure in Streamlit
    st.plotly_chart(fig)
    # MAP OF CAPACITY INSTALLED START

    # FULL SOLAR PLOT START 
    # create generation prediction values
    X_full = pd.concat([X_train, X_test]).sort_index() # Combine X_train and X_test to predict for the whole dataset
    y_index = X_full.index
    y_pred_full = pipeline.predict(X_full)  # Get predictions from the fitted pipeline (preserves order of X_full)

    df_plot = gen_weather_merged_df.copy() # copy to avoid mutation
    df_plot['datetime_gmt'] = pd.to_datetime(df_plot.index)
    df_plot['pred_generation_mw'] = np.nan # Create a column aligned to the feature index with predictions (NaN elsewhere)
    df_plot.loc[y_index, 'pred_generation_mw'] = y_pred_full

    fig = go.Figure()    # Create Plotly figure

    # Actual generation on secondary y-axis
    fig.add_trace(go.Scatter(
        x=df_plot['datetime_gmt'],
        y=df_plot['generation_mw'],
        mode='lines',
        name='Generation (Actual)',
        line=dict(color='blue'),
        yaxis='y2'
    ))

    # Predicted generation on secondary y-axis (use dashed line or different color)
    fig.add_trace(go.Scatter(
        x=df_plot['datetime_gmt'],
        y=df_plot['pred_generation_mw'],
        mode='lines',
        name='Generation (Predicted)',
        line=dict(color='orange'),
        yaxis='y2'
    ))

    # Layout with secondary y-axis
    fig.update_layout(
        title="Solar Generation and Radiation Over Time (Actual vs Predicted)",
        xaxis_title='Date and Time',
        yaxis_title='Radiation Values',
        yaxis2=dict(
            title='Generation (MW)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(rangeslider=dict(visible=True)),
        legend=dict(x=0.01, y=0.99),
        height=600
    )

    st.plotly_chart(fig) # show fig
    # FULL SOLAR PLOT END

    # METRICS TABLE START

    # Actual generation vs. predicted gernation for gsp in selected period
    # variables metrics
    
    gen_series = pd.to_numeric(gen_weather_merged_df['generation_mw'], errors='coerce')
    total_generation_mw = float(gen_series.sum())
    total_predicted_generation_mw = float(np.nansum(y_pred_full))

    if total_generation_mw == 0:
        percent_variance = None  # or float('inf') if you prefer
    else:
        percent_variance = (total_predicted_generation_mw - total_generation_mw) / total_generation_mw * 100.0
    
    cap_series = pd.to_numeric(gen_weather_merged_df['capacity_mwp'], errors='coerce')
    last_capacity_mwp = float(cap_series.iloc[-50])

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generation (MW)", f"{total_generation_mw:.2f}")
    col2.metric("Predicted Generation (MW)", f"{total_predicted_generation_mw:.2f}")
    col3.metric("Variance (%)", f"{percent_variance:.2f}%")
    col4.metric("Latest Installed Capacity (MW)", f"{last_capacity_mwp:.2f}")

    with st.expander("Show generation and capacity data"):
        st.dataframe(generation_and_capacity_single_gsp)

    # METRICS END

    # HEATMAP START

    # Calculate the difference between predicted and actual generation
    df_plot['difference'] = df_plot['pred_generation_mw'] - df_plot['generation_mw']

    # Reset index for heatmap plotting
    heatmap_data = df_plot.reset_index()

    fig = px.density_heatmap(heatmap_data, 
                            x='datetime_gmt', 
                            y='difference',
                            z='difference',
                            color_continuous_scale='RdBu',
                            title='Heatmap of Differences (Predicted - Actual)')

    fig.update_layout(
        xaxis_title='Date and Time',
        yaxis_title='Difference in Generation (MW)',
        height=600
    )

    st.plotly_chart(fig)
    # HEATMAP END

#=============================================================================================
# Tab 1 END

# Tab 2 START
#=============================================================================================
with tab2:

    st.subheader("Selected GSP Stats")



#=============================================================================================
# Tab 2 END

# Footer
st.markdown("---")
st.markdown("**Contact samhsforrest@gmail.com for any questions)")