# ML environments
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import pytz

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

# Set GSP and timeframes to call functions
start = datetime(2025, 1, 1, 0, 0, tzinfo=pytz.UTC) # set later via table
end = datetime(2025, 12, 31, 23, 30, tzinfo=pytz.UTC) # set later via table
gsp_id = 124  # select GSP ID to extract data for - later we will use table of names for this

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

# Usage
pipeline, X_train, X_test, y_train, y_test = prepare_and_fit_xgboost(gen_weather_merged_df)
