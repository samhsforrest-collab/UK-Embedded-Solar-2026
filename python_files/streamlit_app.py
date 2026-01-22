# Environments START
#=============================================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import streamlit as st

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

# Create variables from the XGBoost model
pipeline, X_train, X_test, y_train, y_test = prepare_and_fit_xgboost(gen_weather_merged_df)

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
   
    # Map
    st.subheader("Selected GSP Stats")

#=============================================================================================
# Tab 2 END

# Footer
st.markdown("---")
st.markdown("**©** Copyright protected by The Big Solar Plot est. 2027 (time-travel is the surest way to be ahead of your time)")