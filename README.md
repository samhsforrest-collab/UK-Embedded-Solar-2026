# UK Embedded Solar Generation

UK Embedded Solar Generation is a data analysis tool exyracting data from the Sheffield Solar PVLive API and Open meteo historical weather API to data from 290 Grid Supply Points (GSP) across the UK and evaluate solar generation at each GSP over a 5 year period to an hourly granularity. It trains a machine learning model to predict the hourly solar generaiton profile at any selected GSP (from the list of 290) based on local weather conditions.

# Business Objectives

- Assess whether we can extract, clean, visualise, and accurately forecast embedded solar generation at GSP level across the UK to meet NESO’s balancing market (BM) requirements.
- Improve network operating efficiency by improving accuracy and reducing safety buffer/redundancy in decision-making to unlock cost savings that are passed through to UK electricity consumers.
- 
___
# Hypothesis
- Can we extract, clean, visualise, and forecast the embedded solar generation for GSPs in the UK with sufficient accuracy to support NESO’s instructions in the BM?
- To validate this we are aiming for a 95% explained variance in the model testing (R2)

# Datasets - The APIs

- Dashboard sources data from two PVLive endpoints (Sheffield Solar):
  - 17 MWp cumulative solar capacity (MWp) via PVLive deployment endpoint from 290 GSPs across the UK.
  - 5 years of historic solar generation (MWh) via PVLive generation endpoint from 290 GSPs across the UK.

- GSP location lookup:
  - GSP geo-locations are sourced from the [NESO](https://www.neso.energy/data-portal/gis-boundaries-gb-grid-supply-points) website CSV and matched to PVLive GSP IDs.

- Weather data:
  - 5 years of historic localised hourly weather/irradiance data is retrieved from Open-Meteo Historical API (queried using the selected GSP's lat/lon).

# Project Plan
- Workflow summary:
  1. Fetch cumulative capacity and historic generation from PVLive.
  2. Match PVLive GSP IDs to NESO-provided geo-location metadata.
  3. Use selected GSP lat/lon to request hourly weather from Open-Meteo.
  4. Merge and align generation, capacity and weather time series.
  5. Train, fit and predict with a machine learning model using the combined solar + weather feature set. 
___

# Dashboard Deployment and Usage
Access the [Dashboard](https://uk-embedded-solar-generation-esmkfgpqjmccxirsh9tgcu.streamlit.app/) here

- Select your chosen GSP and date range on the sidebar to train the model, predict and visualise your data below
- Interact with Plotly Graphs using buttons in top right corner of each to zoom and pan
- Dowload generation and capacity data for selected GSP using the dropdown 'Show generation and capacity data' and clicking the download button in the top right hand corner
- Navigate to the second tab to explore the GSP embedded solar map and the history of solar deployment across the UK
___

# Extraction and EDA Functions explained

## load_mwp(region="gsp", include_history=True)
- Loads MWp deployment table from PVLive (pvl.deployment).
- Returns: `mwp_df` (rows = deployments; key cols include `GSPs`, `install_month`, `cumul_capacity_mwp`).

## load_gsp(gsp_path="../data/gsp_info.csv")
- Loads GSP metadata CSV, coerces/filters `gsp_id` to PVLive-known IDs.
- Returns: `gsp_df` (cols include `gsp_id`, `gsp_name`, `gsp_lat`, `gsp_lon`, `region_name`, `pes_id`).

## merge_gsp_location(mwp_df, gsp_df, gsp_col_mwp='GSPs', gsp_col_gsp='gsp_name')
- Case-insensitive join of `mwp_df` to `gsp_df` to add `gsp_id`, lat/lon, `region_name`, `pes_id`.
- Drops rows with missing values and typo `'unkown'`.
- Returns: `merged_df` (capacity rows enriched with location metadata).

## gsp_locations(merged_df, gsp_col='GSPs')
- Produce one-row-per-GSP lookup with location and region info.
- Adds `GSP_region` = `"GSPs | region_name"`.
- Returns: `gsp_locations_list` (cols: `GSPs`, `gsp_id`, `gsp_lat`, `gsp_lon`, `region_name`, `pes_id`, `GSP_region`).

## wide_cumul_capacity(merged_df, time_col='install_month', gsp_col='GSPs', value_col='cumul_capacity_mwp')
- Pivot `merged_df` to wide time series: rows = `install_month`, columns = GSP identifiers, values = cumulative capacity.
- Returns: `capacity_growth_all_gsps` (time column + one column per GSP with `cumul_capacity_mwp`).

## download_generation_for_single_gsp(start, end, gsp_id, gsp_locations_list, ...)
- Fetch half-hourly generation for a single GSP from PVLive; validate `gsp_id`.
- Interpolates short gaps (limit=12 half-hour slots) and merges GSP metadata.
- Returns: `generation_df` (`datetime_gmt`, `gsp_id`, `generation_mw`, plus location/region cols).

## get_capacity_data_single_gsp(gsp_id, merged_df)
- Extract monthly cumulative capacity for a single GSP and derive `month`/`year`/`day`/`hour` columns.
- Returns: `capacity_data` (`install_month`, `cumul_capacity_mwp`, `GSPs`, `gsp_lat`, `gsp_lon`, `region_name`, `pes_id`, `month`, `year`, `day`, `hour`).

## add_capacity_to_generation(generation_df, capacity_data, tz='UTC', gen_fill_method='interpolate')
- Merge monthly capacity into half-hourly generation, forward-fill capacity, reindex to continuous 30-min UTC grid.
- Fill/interpolate generation and capacity; forward-fill metadata columns; recompute time components.
- Returns: `generation_and_capacity_single_gsp` (continuous 30-min series with `generation_mw`, `capacity_mwp`, metadata, `month`/`year`/`day`/`hour`).

## fetch_weather_for_generation_df(gen_df, cache_path=".cache", ...)
- Derive lat/lon and date range from `gen_df`; request hourly irradiance/weather from Open-Meteo (cached + retried).
- Returns: `weather_df` (hourly UTC index, irradiance vars, `year`, `month`, `day`, `hour`).

## merge_generation_with_weather(generation_df, weather_df)
- Aggregate half-hourly generation to hourly (mean), merge with hourly weather on `year`/`month`/`day`/`hour`.
- Interpolates remaining NaNs in generation and capacity; restores datetime index.
- Returns: `gen_weather_merged_df` (hourly weather vars + `generation_mw` + `capacity_mwp`), ready for modeling.

___

# Conclusion: further refinement is required to accurately support NESO’s BM actions

## Successes
- 290 GSPs extracted, cleaned and aligned with weather data.  
- ~17 GWp of solar capacity data available for evaluation.  
- 5 years of historical weather and solar data (17GWp) available for each of the 290 GSPs' locations.  
- 5 ML models trained and tested; winning model achieved ~80% explained variance (R²).

## Data cleaning adjustments
- Reduce interpolation window for generation to avoid skewed profiles; impute remaining NaNs with zeros.  
- Add more comments and tidy Python files for maintainability.  
- Refactor Streamlit app functions to remove interdependencies so they can be moved to functions.py and used with date/GSP sidebar selectors without circular references.

## Model refinements and bug fixes
- Re-tune XGBoost to increase weight on capacity features and account for MWp growth over time.  
- Retrain models to ensure predicted generation is zero during hours of zero irradiance.
- Retrain the model to explore why there appears to be a cap on the predicted solar generation (first page first visual)
- Incorporate additional weather parameters (humidity, wind, temperature).  
- Engage Sheffield Solar API team to investigate access to unique solar project locations.
___

# Development Roadmap & Main Challenges

The complete data extraction and EDA journey can be found in the Jupyter Notebook files which should be read in this order:

1. workings_pvlive_api_mwp_gsp_mapping_functions_1.ipynb
2. workings_pvlive_api_generation_mwp_functions_2.ipynb
3. workings_openmeteo_weather_api_and_pvlive_functions_3.ipynb
4. workings_openmeteo_and_pvlive_continued_4.ipynb
5. workings_ml_predictions_functions_5.ipynb

The main challenge not shown in the above workings happened when migrating the functions from the Jupyter Notebooks into separate py files for use in the streamlit app.
- The intention was to create separate files for cleanliness and code visibility
- However the functions had been built with interdependencies between them (variable names required by the next function below). This wasn't spotted in the notebooks given the functions worked because the variables were saved in the kernel during the buidling process
- When migrating to the python files and building in datetime and GSP selection functions, the interdependency between the above functions created a circular reference between the files.
- The only way to resolve this in time for this project was to build the affecting functions directly into the streamlit_app.py file, however this is messy and poor for code visibility.
- Next time I would build the functions without these interdependencies so they can each stand alone and be moved to the functions.py file (and used with sidebar selectors) without causing a circular reference.
___

# Ethical Considerations

- The Sheffield Solar and Openmeteo APIs were both created for public use.
- There is no personal data involved so GDPR need not be considered.
___

# Credits

This project uses teaching materials from the **Code Institute – Data Analytics with AI Bootcamp**, which supported:

- Data cleaning  
- Feature engineering  
- Exploratory analysis
- Machine learning model scoring code blocks
- Visualisations

Chat GPT and Copilot were used as code assistants for:

- Help write short code blocks
- Troubleshooting code problems
- Checking overall logic and bouncing/genrating ideas for deepening analysis
- Converting writen Readme (word file) into code format
- Explaining logic within API endpoints

[Open meteo API](https://open-meteo.com/en/docs/historical-weather-api)
[Sheffield Solar API Github](https://github.com/SheffieldSolar/PV_Live-API)
