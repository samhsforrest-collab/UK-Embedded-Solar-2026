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
st.image("assets/solar-image2.jpg", width='stretch')
st.sidebar.image("assets/logo.png", width='stretch')

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
default_start_date= datetime(2025, 1, 1).date()  # Converts it to a date object
default_end_date= datetime(2025, 12, 31).date()  # Converts it to a date object

start_date = st.sidebar.date_input(
    "Start Date",  # Label for the date input
    value=default_start_date,  # Default start date
    min_value=min_start_date,  # Minimum allowed date
    max_value=datetime.today().date(),  # Maximum allowed date
    format="YYYY/MM/DD",  # Date format
    disabled=False,  # Whether the input should be disabled
    label_visibility="visible",  # Visibility of the label
    width="stretch"  # Width of the input element
)
end_date = st.sidebar.date_input(
    "End Date",  # Label for the date input
    value=default_end_date,  # Default end date
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
# find gsp_id using get_gsp_id_from_region function for given sidebar GSP_region selection 
gsp_id = get_gsp_id_from_region(gsp_locations_list, selected_gsp)

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

@st.cache_resource
def cached_train(df, target_col='generation_mw', test_size=0.2, random_state=101):
    return prepare_and_fit_xgboost(df, target_col=target_col, test_size=test_size, random_state=random_state)

# Clear cached model helper (optional button to retrain)
if st.sidebar.button("Retrain (clear cache)"):
    try:
        st.cache_resource.clear()
    except Exception:
        # fallback for older streamlit versions
        st.session_state.pop("trained_model", None)
    st.experimental_rerun()

# Train only when button pressed
if "trained_model" not in st.session_state:
    if st.button("Train model"):
        with st.spinner("Training model..."):
            pipeline, X_train, X_test, y_train, y_test = cached_train(gen_weather_merged_df)
            st.session_state["trained_model"] = {
                "pipeline": pipeline,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
            st.success("Model trained.")
else:
    # model already trained (from cache/session)
    stored = st.session_state["trained_model"]
    pipeline = stored["pipeline"]
    X_train = stored["X_train"]
    X_test = stored["X_test"]
    y_train = stored["y_train"]
    y_test = stored["y_test"]

# Only run predictions/plot if pipeline exists
if "pipeline" in locals():
    X_full = pd.concat([X_train, X_test]).sort_index()
    y_index = X_full.index
    y_pred_full = pipeline.predict(X_full)

    df_plot = gen_weather_merged_df.copy()
    df_plot['datetime_gmt'] = pd.to_datetime(df_plot.index)
    df_plot['pred_generation_mw'] = np.nan
    df_plot.loc[y_index, 'pred_generation_mw'] = y_pred_full

    # ... (plotting code unchanged)


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
    st.info(" Select your GSP and date range to train the model, predict and visualise your data below")
    last_row = capacity_growth_all_gsps.iloc[-1]  # Last row values

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
        title="Predicted Solar Generation Vs. Actual)",
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
    st.info("Use the slider bar above to view granluar data periods")
    # METRICS TABLE START

    # Actual generation vs. predicted generation for gsp in selected period
    
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
    st.markdown("---")
    # MONTH SOLAR PLOT
    # align into gen_weather_merged_df
    gen_weather_merged_df['pred_generation_mw'] = np.nan
    gen_weather_merged_df.loc[X_full.index, 'pred_generation_mw'] = y_pred_full

    def plot_last_month_clustered_gen_vs_irradiance(df, pipeline=None, X_full=None):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # ensure predicted column exists: if not, and pipeline+X_full provided, compute predictions
        if 'pred_generation_mw' not in df.columns:
            if pipeline is None or X_full is None:
                raise ValueError("pred_generation_mw missing; provide pipeline and X_full to compute predictions.")
            # predict and align
            y_pred_full = pipeline.predict(X_full)
            pred_series = pd.Series(np.nan, index=df.index)
            pred_series.loc[X_full.index] = y_pred_full
            df['pred_generation_mw'] = pred_series

        # select last month
        last_ts = df.index.max()
        last_year, last_month = last_ts.year, last_ts.month
        df_month = df[(df.index.year == last_year) & (df.index.month == last_month)]

        # daily aggregates
        daily_actual = df_month['generation_mw'].resample('D').sum()
        daily_pred = df_month['pred_generation_mw'].resample('D').sum()
        daily_irr = df_month['global_tilted_irradiance_instant'].resample('D').mean()

        plot_df = pd.DataFrame({
            'date': daily_actual.index,
            'actual_generation_mw': daily_actual.values,
            'pred_generation_mw': daily_pred.values,
            'global_tilted_irradiance_instant': daily_irr.values
        })

        # clustered bars + line
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_df['date'],
            y=plot_df['actual_generation_mw'],
            name='Actual Generation',
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            x=plot_df['date'],
            y=plot_df['pred_generation_mw'],
            name='Predicted Generation',
            marker_color='#FF7A00'
        ))
        fig.add_trace(go.Scatter(
            x=plot_df['date'],
            y=plot_df['global_tilted_irradiance_instant'],
            name='Global Tilted Irradiance (instant)',
            mode='lines+markers',
            marker=dict(color='lightgray'),
            yaxis='y2'
        ))

        fig.update_layout(
            barmode='group',
            title=f"Daily Generation and Irradiance Profile — {last_year}-{last_month:02d}",
            xaxis=dict(title='Date'),
            yaxis=dict(title='Daily Generation (MW total)'),
            yaxis2=dict(title='Irradiance (W/m²)', overlaying='y', side='right', showgrid=False),
            legend=dict(x=0.01, y=0.99),
            height=520
        )

        st.plotly_chart(fig, width='stretch')

    # Usage:
    
        # Layout: first row -> left placeholder, right line plot
    col1, col2 = st.columns([1,2])  # right column wider
    with col1:
        st.markdown("""\n\n
               
**Daily Solar Generation vs. Predictions (last month selected):**

 We note predictions tend to track irradiance more closely
than actual generation. Reasons may include: 

1) predictions are based on a single location (and weather conditions) at the GSP, 
but the solar plants are scattered in multiple locations, and;

2) generation may have been affected by other factors such as grid 
and/or localised plant faults.
""")
    with col2:
        plot_last_month_clustered_gen_vs_irradiance(gen_weather_merged_df)

    
    # PRIOR MONTH SOLAR PLOT END

    def plot_last_day_hourly_gen_vs_irradiance(df, pipeline=None, X_full=None):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # ensure pred column exists or compute from pipeline+X_full
        if 'pred_generation_mw' not in df.columns:
            if pipeline is None or X_full is None:
                raise ValueError("pred_generation_mw missing; provide pipeline and X_full to compute predictions.")
            y_pred_full = pipeline.predict(X_full)
            pred_series = pd.Series(np.nan, index=df.index)
            pred_series.loc[X_full.index] = y_pred_full
            df['pred_generation_mw'] = pred_series

        # select last day present in dataframe
        last_ts = df.index.max()
        last_day = last_ts.date()
        df_day = df[df.index.date == last_day]

        # if no rows for last_day, try previous day
        if df_day.empty:
            last_day = (last_ts - pd.Timedelta(days=1)).date()
            df_day = df[df.index.date == last_day]

        # aggregate hourly (if already hourly, this keeps per-hour values)
        hourly_actual = df_day['generation_mw'].resample('h').sum()
        hourly_pred = df_day['pred_generation_mw'].resample('h').sum()
        hourly_irr = df_day['global_tilted_irradiance_instant'].resample('h').mean()

        plot_df = pd.DataFrame({
            'hour': hourly_actual.index,
            'actual_generation_mw': hourly_actual.values,
            'pred_generation_mw': hourly_pred.values,
            'global_tilted_irradiance_instant': hourly_irr.values
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_df['hour'],
            y=plot_df['actual_generation_mw'],
            name='Actual Generation',
            marker_color='#1f77b4'
        ))
        fig.add_trace(go.Bar(
            x=plot_df['hour'],
            y=plot_df['pred_generation_mw'],
            name='Predicted Generation',
            marker_color='#FF7A00'
        ))
        fig.add_trace(go.Scatter(
            x=plot_df['hour'],
            y=plot_df['global_tilted_irradiance_instant'],
            name='Global Tilted Irradiance (instant)',
            mode='lines+markers',
            marker=dict(color='lightgray'),
            yaxis='y2'
        ))

        fig.update_layout(
            barmode='group',
            title=f"Hourly Generation and Irradiance Profile — {last_day}",
            xaxis=dict(title='Hour', tickformat='%H:%M'),
            yaxis=dict(title='Generation (MW)'),
            yaxis2=dict(title='Irradiance (W/m²)', overlaying='y', side='right', showgrid=False),
            legend=dict(x=0.01, y=0.99),
            height=520
        )

        st.plotly_chart(fig, width='stretch')

    
    # Second row -> left bar plot, right placeholder
    col3, col4 = st.columns([2,1])  # left wider this row
    with col3:
        plot_last_day_hourly_gen_vs_irradiance(gen_weather_merged_df)
    with col4:
        st.markdown("""
**Hourly Actual vs. Predicted Generation (last day selected):**

This plot highlights required adjustments to improve accuracy in 
the next iteration of the ML model:

1) The XGBoost ML model predicts some small generation during hours of 0 irradiance, we
will adjust to ensure 0 generation during these hours.

2) The interpolation method extends too far when imputing missing actual generation data,
we will reduce the scope of the interpolation and impute remainder of NaNs with 0s.
                    """
        )
    # Or provide pipeline and X_full to compute predictions inside the function:
    # plot_last_day_hourly_gen_vs_irradiance(gen_weather_merged_df, pipeline=pipeline, X_full=X_full)


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
                            title='Heatmap of Differences (Predicted vs. Actual Generation)')

    fig.update_layout(
        xaxis_title='Date and Time',
        yaxis_title='Difference in Generation (MW)',
        height=600
    )

    st.plotly_chart(fig)
    
    st.markdown("""**Heatmap Analysis:**

The Heatmap shows the model becomes increasingly inaccurate over time as the solar capacity increases. We suspect
this is likely due to an incorrect weighting of the cumulative capacity feature in the dataset.
Solution: heavily weight the GSPs solar capacity feature when fine-tuning of the next iteration of the ML model

""")
    # HEATMAP END

#=============================================================================================
# Tab 1 END

# Tab 2 START
#=============================================================================================
with tab2:

    # Headings
    st.subheader("UK Embedded Solar by GSP")
    
    # Capacity and GSP counts
    gsp_count = len([c for c in capacity_growth_all_gsps.columns if c != 'install_month']) # Count columns excluding 'install_month' (index is not a column)
    total_solar_capacity = int(capacity_growth_all_gsps.iloc[-1].iloc[2:].sum()/1000)  # Sum values in the last row from the 3rd column (index 2) onwards
    st.info(f"This dashboard plots {gsp_count} GSPs with a combined total of {total_solar_capacity}GWp solar capacity")

    # MAP OF CAPACITY INSTALLED START
    # Create a Capacity DataFrame where each GSP gets the last capacity values.
    latest_capacity_dict = {
        'GSPs': gsp_locations_list['GSPs'],  # GSP identifiers
        'Latest Installed Capacity (MW)': [last_row[gsp] for gsp in gsp_locations_list['GSPs']]  # GSPs must match column names
    }
    latest_capacity_df = pd.DataFrame(latest_capacity_dict)
    gsp_locations_with_capacity = gsp_locations_list.merge(latest_capacity_df, on='GSPs', how='left')

    fig = px.scatter_map( # Create a scatter map with sizes based on Latest Installed Capacity
        gsp_locations_with_capacity,
        lat="gsp_lat",
        lon="gsp_lon",
        hover_name="region_name",  # Show region name on hover
        size="Latest Installed Capacity (MW)",  # Scale point size based on this capacity
        size_max=20,  # Max size for points
        title="GSP Locations showing Latest Installed Solar Capacity (MWp)",
        map_style="open-street-map",
        hover_data={"Latest Installed Capacity (MW)": ":.0f"}  # format as integer
    )

    # Show the figure in Streamlit
    st.plotly_chart(fig)
    # MAP OF CAPACITY INSTALLED END

    # LINE PLOT SHOWING CAPACITY GROWTH START 
    # Example: create line_fig and bar_fig using your helper functions (replace with your actual code)
    def make_line_fig(capacity_growth_all_gsps):
        df = capacity_growth_all_gsps.copy()
        df['install_month'] = pd.to_datetime(df['install_month'])
        df = df.set_index('install_month').sort_index()
        df = df.loc[df.index >= pd.Timestamp("2010-01-01")]
        gsp_cols = [c for c in df.columns if c != 'install_month']
        # limit to top N to keep plot readable
        top_n = 50
        top_cols = df[gsp_cols].iloc[-1].nlargest(top_n).index.tolist()
        fig = px.line(df.reset_index(), x='install_month', y=top_cols, title="Cumulative capacity over time by GSP (2010 onwards)")
        fig.update_traces(hoverinfo='skip')  # remove hover data
        fig.update_layout(showlegend=False, height=550, xaxis_title='Install Month', yaxis_title='Cumulative MWp')
        return fig

     # LINE PLOT SHOWING CAPACITY GROWTH END
    
     #  BAR PLOT SHOWING CAPACITY INSTALLED BY YEAR START
    def make_bar_fig(capacity_growth_all_gsps):
        df = capacity_growth_all_gsps.copy()
        df['install_month'] = pd.to_datetime(df['install_month'])
        df = df.set_index('install_month').sort_index()
        df = df.loc[df.index >= pd.Timestamp("2009-01-01")]
        gsp_cols = [c for c in df.columns if c != 'install_month']
        annual_cumul = df[gsp_cols].resample('Y').last()
        annual_total = annual_cumul.sum(axis=1)
        annual_increase = annual_total.diff().fillna(0)
        plot_df = pd.DataFrame({
            'year': annual_increase.index.year,
            'annual_increase_mwp': annual_increase.values,
            'total_capacity_mwp': annual_total.values
        })
        fig = px.bar(
            plot_df,
            x='year',
            y='annual_increase_mwp',
            labels={'annual_increase_mwp': 'Capacity increase (MWp)', 'year': 'Year'},
            text='annual_increase_mwp',
            hover_data={'annual_increase_mwp': ':.0f', 'total_capacity_mwp': ':.0f'}
        )
        fig.update_traces(marker_color='#FF7A00', texttemplate='%{text:.0f}', textposition='outside')
        fig.update_layout(title='Annual Capacity Increase (MWp)', xaxis=dict(tickmode='linear'), yaxis_title='MWp', height=450)
        return fig
     #  BAR PLOT SHOWING CAPACITY INSTALLED BY YEAR END

    line_fig = make_line_fig(capacity_growth_all_gsps)
    bar_fig  = make_bar_fig(capacity_growth_all_gsps)

    # Layout: first row -> left placeholder, right line plot
    col1, col2 = st.columns([1,2])  # right column wider
    with col1:
        st.markdown("""**UK Embedded Solar Capacity Over Time:**

Significant growth of solar capacity over the past 15 years was driven
initially by the Feed-in-Tariff (FiT) subsidy scheme from 2010 to 2015 with later growth marked
by the reducing cost of building and maintaining solar, making it now
one of the cheapest forms of producing electricity. The latest challenge faced by the industry
is now caused by an over-congested UK electricity grid network. Particularly for larger installations which are stuggling to find capacity on a network which
was initially designed to operate with a few large-scale base-load turbines connected to
Transmission Networks, not at Distribution level (i.e, embedded))
""")
    with col2:
        st.plotly_chart(line_fig, width='stretch')

    # Second row -> left bar plot, right placeholder
    col3, col4 = st.columns([2,1])  # left wider this row
    with col3:
        st.plotly_chart(bar_fig, width='stretch')
    with col4:
        st.markdown("""**Bar Graph Showing New Installed Solar Capacity by Year:**

Major capacity gains were initially seen in 2011 following the launch of the FiT scheme (2010)
with a slight reduction in 2012 caused by reducing FiT support rates, but further growth (up to 2015) seen
as economies of scale drove down manufacturing costs marking viability for larger projects supported by the Renewable
Obligation Certification (ROC). ROCs, however, were subsequently restricted to only new solar projects below 5MW in 2015. From 2020 onwards capacity growth is
purely subsidy-free given the complete closure of the FiT and ROC schemes for all new solar in 2019 and 2017, respectively.
                    """
        )

#=============================================================================================
# Tab 2 END

# Footer
st.markdown("---")
st.markdown("**Contact samhsforrest@gmail.com for any questions)")