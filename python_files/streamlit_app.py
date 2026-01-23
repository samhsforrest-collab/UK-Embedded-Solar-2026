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

import pytz
from datetime import datetime
import time
from datetime import datetime, timedelta, timezone
#=============================================================================================
# Environments END

# Extra Functions START
#=============================================================================================

# Pull functions and associated dataframes from pvlive_openmeteo
from pvlive_openmeteo_api_functions import (
    load_mwp,
    load_gsp,
    merge_gsp_location,
    gsp_locations,
    wide_cumul_capacity,
    download_generation_for_single_gsp,
    get_capacity_data_single_gsp,
    add_capacity_to_generation,
    fetch_weather_for_generation_df,
    merge_generation_with_weather
)
# Pull winning XGBoost ML model
from xgboost_model import prepare_and_fit_xgboost

# Set GSP and timeframes to call functions
start = datetime(2025, 1, 1, 0, 0, tzinfo=pytz.UTC)
end = datetime(2025, 12, 31, 23, 30, tzinfo=pytz.UTC)
gsp_id = 124  # select GSP ID to extract data for - later we will ue table of names for this


# Call functions and create variables holidng dataframes for use in the app
mwp_df = load_mwp() # loading the capacity df
gsp_df = load_gsp() # loading the gsp locations df
merged_df = merge_gsp_location(mwp_df, gsp_df) # merge capacity growth and locations
gsp_locations_list = gsp_locations(merged_df) # merge capacity and locations without capacity growth over time
generation_df = download_generation_for_single_gsp(start, end, gsp_id, gsp_locations_list) # generation df for selected gsp
capacity_growth_all_gsps = wide_cumul_capacity(merged_df) # wide capacity growth df for all time and all gsps
capacity_data_single_gsp = get_capacity_data_single_gsp(gsp_id, merged_df) # add month and year to capacity single gsp
generation_and_capacity_single_gsp = add_capacity_to_generation(generation_df, capacity_data_single_gsp) # merged capacity and generation same time-series single gsp
weather_df = fetch_weather_for_generation_df(generation_and_capacity_single_gsp) # weather df for the selected gsp and time period
gen_weather_merged_df = merge_generation_with_weather(generation_and_capacity_single_gsp, weather_df)

# Create variables from the XGBoost model
pipeline, X_train, X_test, y_train, y_test = prepare_and_fit_xgboost(gen_weather_merged_df)

#=============================================================================================
# Extra Functions END

# Create tabs START
#=============================================================================================
tab1, tab2 = st.tabs(["UK Embedded Solar Generation", "Selected GSP"])
#=============================================================================================
# Create tabs END

# Tab 1 START
#=============================================================================================
with tab1:
   
    # Map
    st.subheader("UK Embedded Solar Generation by GSP")

#=============================================================================================
# Tab 1 END

# Tab 2 START
#=============================================================================================
with tab2:
    
    st.subheader("Selected GSP Stats")
    # Assume: pipeline is fitted, X_train/X_test are the feature DataFrames used for training/testing,
    # and gen_weather_merged_df contains the 'generation_mw' actuals with a datetime index.

    # Combine X_train and X_test to predict for the whole dataset you're evaluating
    X_full = pd.concat([X_train, X_test]).sort_index()
    y_index = X_full.index

    # Get predictions from the fitted pipeline (preserves order of X_full)
    y_pred_full = pipeline.predict(X_full)

    # Prepare a copy of the original df for plotting (avoid mutation)
    df_plot = gen_weather_merged_df.copy()
    df_plot['datetime_gmt'] = pd.to_datetime(df_plot.index)

    # Create a column aligned to the feature index with predictions (NaN elsewhere)
    df_plot['pred_generation_mw'] = np.nan
    df_plot.loc[y_index, 'pred_generation_mw'] = y_pred_full

    # Create Plotly figure with the same radiation traces plus actual and predicted generation
    fig = go.Figure()

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
        title='Solar Generation and Radiation Over Time (Actual vs Predicted)',
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

    st.plotly_chart(fig)


#=============================================================================================
# Tab 2 END

# Footer
st.markdown("---")
st.markdown("**©** Copyright protected by The Big Solar Plot est. 2027 (time-travel is the surest way to be ahead of your time)")