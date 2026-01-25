# Business Objectives

- Assess whether we can extract, clean, visualise, and accurately forecast embedded solar generation at GSP level across the UK to meet NESO’s balancing market (BM) requirements.
- Improve network operating efficiency by improving accuracy and reducing safety buffer/redundancy in decision-making to unlock cost savings that are passed through to UK electricity consumers.

___

# APIs

- Dashboard sources data from two PVLive endpoints (Sheffield Solar):
  - Cumulative solar capacity (MWp) via PVLive deployment endpoint.
  - Historic solar generation (MWh) via PVLive generation endpoint.

- GSP location lookup:
  - GSP geo-locations are sourced from the NESO website CSV and matched to PVLive GSP IDs.

- Weather data:
  - Localised hourly historical weather/irradiance is retrieved from Open-Meteo Historical API (queried using the selected GSP's lat/lon).

- Workflow summary:
  1. Fetch cumulative capacity and historic generation from PVLive.
  2. Match PVLive GSP IDs to NESO-provided geo-location metadata.
  3. Use selected GSP lat/lon to request hourly weather from Open-Meteo.
  4. Merge and align generation, capacity and weather time series.
  5. Train, fit and predict with a machine learning model using the combined solar + weather feature set.
  6. 
___

# Main Functions and Feature Engineering

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
- ~17 GWp of solar capacity available for evaluation.  
- 5 years of historical weather and solar data available for analysis.  
- 5 ML models trained and tested; winning model achieved ~80% explained variance (R²).

## Data cleaning adjustments
- Reduce interpolation window for generation to avoid skewed profiles; impute remaining NaNs with zeros.  
- Add more comments and tidy Python files for maintainability.  
- Refactor Streamlit app functions to remove interdependencies so they can be moved to functions.py and used with date/GSP sidebar selectors without circular references.

## Model refinement adjustments
- Re-tune XGBoost to increase weight on capacity features and account for MWp growth over time.  
- Retrain models to ensure predicted generation is zero during hours of zero irradiance.  
- Incorporate additional weather parameters (humidity, wind, temperature).  
- Engage Sheffield Solar API team to investigate access to unique solar project locations.

