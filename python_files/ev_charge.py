# Environments START
# #=============================================================================================
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
import base64
import pydeck as pdk
from pathlib import Path
import matplotlib.pyplot as plt

#=============================================================================================
# Environments END

# Loading images and GIFs START 
#=============================================================================================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

banner_b64 = get_base64_image("ev_charge2.png")

st.sidebar.image("logo.png", use_container_width=True)

st.markdown(
    f"""
    <style>
    .image-container {{
        position: relative;
        width: 100%;
    }}

    .banner-img {{
        width: 100%;
        display: block;
    }}

    </style>

    <div class="image-container">
        <img src="data:image/jpg;base64,{banner_b64}" class="banner-img">
    </div>
    """,
    unsafe_allow_html=True
)
#=============================================================================================
#Images and GIFs END 

# Page configurations and settings START
#=============================================================================================
st.set_page_config(page_title="EV Analysis", page_icon="🚗", layout="wide")

st.markdown(
    """
    <h1 style="color:#007700; font-weight:bold;">
        Electric Vehicles (EVs) and Charge Point Analysis
    </h1>
    <p style="color:#007700; font-weight:bold; font-size:1.1rem;">
        We analyse over 1,000 charge points and c.500 EV car models across 5 major US cities to help you plot
        your next road trip with the nearest, cheapest and fastest EV charge points and top-rated electric wagons!
    </p>
    """,
    unsafe_allow_html=True
)
# Custom styling
page_style = """
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #145A32 !important;
}

/* Sidebar inner content */
[data-testid="stSidebar"] > div:first-child {
    background-color: #007700 !important;
}

/* Sidebar text color */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main page background */
[data-testid="stAppViewContainer"] {
    background-color: #E9FFE9 !important;
}

</style>

"""
st.markdown(page_style, unsafe_allow_html=True)
#=============================================================================================
# Page settings END 

#=============================================================================================
# Loading datasets START

# Load dataset 1
@st.cache_data
def load_data():
    df = pd.read_csv("ev_charging_patterns.csv")
    return df

df = load_data()

# Load dataset 2
@st.cache_data
def load_market():
    df_market = pd.read_csv("electric_vehicles_spec_2025.csv")
    return df_market

df_market = load_market()

# Loading mapping datasets and Geoson  (3)
@st.cache_data
def load_stations():
    df3 = pd.read_csv('stations_with_land_only_coords.csv')

    required_cols = [
        "Charging Station Location",
        "Charger Type",
        "lat",
        "lon",
    ]

    # Filter to target cities
    target_cities = [
    "Houston",
    "San Francisco",
    "Los Angeles",
    "Chicago",
    "New York",
]
    
    df3 = df3[df3["Charging Station Location"].isin(target_cities)]
    # Drop rows without coords
    df3 = df3.dropna(subset=["lat", "lon"])
    return df3

@st.cache_data
def load_city_geojson():
    import requests

    STATE_GEOJSON_URLS = {
        "CA": "https://raw.githubusercontent.com/generalpiston/geojson-us-city-boundaries/master/states/ca.json",
        "NY": "https://raw.githubusercontent.com/generalpiston/geojson-us-city-boundaries/master/states/ny.json",
        "IL": "https://raw.githubusercontent.com/generalpiston/geojson-us-city-boundaries/master/states/il.json",
        "TX": "https://raw.githubusercontent.com/generalpiston/geojson-us-city-boundaries/master/states/tx.json",
    }

    CITIES_BY_STATE = {
        "CA": {"San Francisco", "Los Angeles"},
        "NY": {"New York"},
        "IL": {"Chicago"},
        "TX": {"Houston"},
    }

    features = []

    for state, url in STATE_GEOJSON_URLS.items():
        data = requests.get(url).json()
        wanted = CITIES_BY_STATE[state]

        for feat in data.get("features", []):
            props = feat.get("properties", {})
            name = props.get("NAME") or props.get("name")
            if name in wanted:
                features.append(feat)

    return {
        "type": "FeatureCollection",
        "features": features,
    }

# Load dataset 4 - additional Map section and hover data
@st.cache_data
def load_info():
    info_df = pd.read_csv('ev_charging_patterns_new.csv')
    return info_df

# Load dataset 5 - Used for Availability and Cars Tabs
@st.cache_data
def load_market():
    df_usage = pd.read_csv('clean_ev_data.csv')
    return df_usage

# Clarifying variables for dfs
info_df=load_info()
city_geojson = load_city_geojson()
df3 = load_stations()
df_usage = load_market()

#=============================================================================================
# Data uploads END

# Sidebar filters START 
#=============================================================================================
st.sidebar.header("Apply filters to explore charge points")

# Sidebar container
with st.sidebar:
    # Use st.radio label instead of separate markdown
    select_city_sidebar = st.radio(
        label="Select City",  # Label is now attached to radio
        options=sorted(df3["Charging Station Location"].unique()),
        index=0,  # default selection
        horizontal=False
    )

    central_frequency = st.sidebar.radio(
        "Select Map Hover Stats",
        options=["Mean", "Median", "Min", "Max"],
        index=0,  # default = Mean Average,
        horizontal=True
    )

    options = ["Frequent User Type", "Charging Duration", "Energy Consumed", "Frequent Vehicle Model"]

    select_info = st.sidebar.pills("Select Map Hover Info", options, selection_mode="multi")
    
select_charger_types_sidebar= st.sidebar.multiselect(
    "Select Charger Types",
    options=sorted(df3["Charger Type"].unique()),
    default=sorted(df3["Charger Type"].unique()),
)

point_size_label_sidebar = st.sidebar.radio(
        "Point Size",
        options=["Very Small", "Small", "Medium", "Large"],
        index=2,  # default = Medium,
        horizontal=True
    )
POINT_SIZE_MAP = {
    "Very Small": 10,
    "Small": 250,
    "Medium": 500,
    "Large": 2000,
}
CENTRAL_TENDENCY_MAP = {
    "Mean": "mean",
    "Median": "median",
    "Min": "min",
    "Max": "max",
}

agg_func = CENTRAL_TENDENCY_MAP[central_frequency]

# Numeric stats: duration + energy per station
numeric_stats = (
    info_df
    .groupby("station_id")[["Charging Duration (hours)", "Energy Consumed (kWh)"]]
    .agg(agg_func)
    .round(2)
)

# Frequent (mode) user type / vehicle model per station
user_mode = (
    info_df
    .groupby("station_id")["User Type"]
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
    .rename("Frequent User Type")
)

vehicle_mode = (
    info_df
    .groupby("station_id")["Vehicle Model"]
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
    .rename("Frequent Vehicle Model")
)

# Combine all stats into one DataFrame
stats_per_station = (
    numeric_stats
    .join(user_mode)
    .join(vehicle_mode)
    .reset_index()
)
point_radius = POINT_SIZE_MAP[point_size_label_sidebar]

df3 = df3.merge(stats_per_station, on="station_id", how="left")

city_df = df3[df3["Charging Station Location"] == select_city_sidebar]

if select_charger_types_sidebar:
    df_filtered = city_df[city_df["Charger Type"].isin(select_charger_types_sidebar)].copy()
else:
    #keep map center from city_df if no charger type is selected
    df_filtered = city_df.iloc[0:0].copy()  # empty frame with same columns
#=============================================================================================
# Sidebar Filters END

# General Feature Engineering START
#=============================================================================================
# df 1
# Create Unique IDs
df["Charging Station Location"] = df["Charging Station Location"].astype(str)
df["User ID"] = df["User ID"].astype(str)
df["Unique ID"] = df["Charging Station Location"] + "_" + df["User ID"]

# Create Unit Cost $/kWh and drop outliers
df['Cost Per kWh ($)'] = df['Charging Cost (USD)'].astype(float)/df['Energy Consumed (kWh)'].astype(float)
# Ensure numeric
df['Cost Per kWh ($)'] = pd.to_numeric(df['Charging Cost (USD)'], errors='coerce') / \
                         pd.to_numeric(df['Energy Consumed (kWh)'], errors='coerce')
# Calculate Q1, Q3 and IQR
Q1 = df['Cost Per kWh ($)'].quantile(0.25)
Q3 = df['Cost Per kWh ($)'].quantile(0.75)
IQR = Q3 - Q1

# Filter out major outliers
df = df[(df['Cost Per kWh ($)'] >= (Q1 - 5 * IQR)) & 
                 (df['Cost Per kWh ($)'] <= (Q3 + 5 * IQR))]

# df 2 (renamed df_market)
# Data Engineering
# Estimate price based on segment (since 2025 data lacks price)
def estimate_price(segment):
    if pd.isna(segment): return 45000
    if 'A -' in segment or 'B -' in segment: return 30000  # Economy
    if 'C -' in segment or 'D -' in segment: return 48000  # Mid-Range
    if 'E -' in segment or 'F -' in segment: return 85000  # Premium
    if 'S -' in segment: return 75000 # Sports
    return 55000 

df_market['Estimated_Price'] = df_market['segment'].apply(estimate_price)

# Simplify segments
def simplify_segment(segment):
    if pd.isna(segment): return 'Mid-Range'
    if 'A -' in segment or 'B -' in segment: return 'Economy'
    if 'E -' in segment or 'F -' in segment or 'S -' in segment: return 'Premium'
    return 'Mid-Range'

df_market['Category'] = df_market['segment'].apply(simplify_segment)

# Standardise Car Body Type 
def simplify_body(body):
    if pd.isna(body): return 'Other'
    if 'SUV' in body: return 'SUV'
    if 'Sedan' in body: return 'Sedan'
    if 'Hatchback' in body: return 'Hatchback'
    return 'Other'

df_market['Body_Type'] = df_market['car_body_type'].apply(simplify_body)

#=============================================================================================
# General Feature Engineering END

# Headline Summary Stats START
#=============================================================================================
with st.container():
    st.subheader(f"Summary Statistics - {select_city_sidebar}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Get filtered values from df_filtered
        selected_cities = df_filtered["Charging Station Location"].unique()
        selected_charger_types = df_filtered["Charger Type"].unique()

        # Apply those filters to the full df
        df_metric = df[
            (df["Charging Station Location"].isin(selected_cities)) &
            (df["Charger Type"].isin(selected_charger_types))
        ]

        # Count unique charging stations
        total_stations = df_metric["Unique ID"].nunique()

        st.metric("Total Charging Stations Selected", total_stations)

    with col2:
        # Calculate average cost per unit for filtered sessions
        avg_cost_per_kwh = df_metric["Cost Per kWh ($)"].mean()
        # Display metric (rounded to 2 decimal places)
        st.metric("Average Cost Per Unit ($/kWh)", f"${avg_cost_per_kwh:.2f}")
    with col3:
        # Calculate average spend per session for filtered data
        avg_spend_per_session = df_metric["Charging Cost (USD)"].mean()
        # Display metric (2 decimal places)
        st.metric("Average Spend Per Session ($)", f"${avg_spend_per_session:.2f}")
    with col4:
        # Calculate average kWh consumed per session for filtered data
        avg_kwh_per_session = df_metric["Energy Consumed (kWh)"].mean()
        # Display metric (2 decimal places)
        st.metric("Average Energy Per Session (kWh)", f"{avg_kwh_per_session:.0f} (kWh)")
#=============================================================================================
# Headline Summary Stats END

# Create tabs START
#=============================================================================================
tab1, tab2, tab3, tab4 = st.tabs(["Map Plot", "Pricing Analysis", "Availability Analysis", "EV Car Market"])
#=============================================================================================
# Create tabs END

# Tab 1 Mapping START
#=============================================================================================
with tab1:
   
    # Map
    st.subheader("Charging Stations Plotted")

    # Background is the US polygons
    geo_layer = pdk.Layer(
        "GeoJsonLayer",
        data=city_geojson,
        pickable=False,
        stroked=True,
        filled=True,
        get_fill_color="[173, 216, 230, 120]",
        get_line_color="[0, 0, 139, 180]",
        line_width_min_pixels=1,
    )

    COLOR_MAP = {
        "DC Fast Charger": [13, 110, 253, 180],  # blue
        "Level 1": [255, 165, 0, 180],           # sun orange
        "Level 2": [34, 139, 34, 180],          # forest green
    }
    df_filtered["color"] = df_filtered["Charger Type"].apply(
        lambda x: COLOR_MAP.get(x, [160, 160, 160, 180])
    )

    station_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_filtered,
        get_position="[lon, lat]",
        get_radius=point_radius,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    # Build dynamic tooltip
    tooltip_html = f"<b>{{Charging Station Location}}</b><br>{{Charger Type}}<br>"
    if "Charging Duration" in select_info:
        tooltip_html += "Charging Duration (hours): {Charging Duration (hours)}<br>"
    if "Energy Consumed" in select_info:
        tooltip_html += "Energy Consumed (kWh): {Energy Consumed (kWh)}<br>"
    if "Frequent User Type" in select_info:
        tooltip_html += "Frequent User Type: {Frequent User Type}<br>"
    if "Frequent Vehicle Model" in select_info:
        tooltip_html += "Frequent Vehicle Model: {Frequent Vehicle Model}<br>"

    tooltip = {
        "html": tooltip_html,
        "style": {
            "backgroundColor": "rgba(0,0,0,0.8)",
            "color": "white",
        },
    }

    if not city_df.empty:
        center_lat = city_df["lat"].mean()
        center_lon = city_df["lon"].mean()
        zoom = 10  # close zoom for a city
    else:
        center_lat = df["lat"].mean()
        center_lon = df["lon"].mean()
        zoom = 3.4

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    # keeps GeoJSON visible
    deck = pdk.Deck(
        layers=[geo_layer, station_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None
    )

    st.pydeck_chart(deck, use_container_width=True)

    # ~~~Map END~~~

    with st.expander("Show station data"):
        st.dataframe(df_filtered, use_container_width=True)

#=============================================================================================
# Tab 1 END

# Tab 2 START
#=============================================================================================
with tab2:
    # Basic info
    st.header("Plot The Cheapest Charge Points")
    st.markdown("""
    The bar chart below shows the **average cost per unit ($/kWh)** by **charger type** and **city**. The points are split into:

    - **DC Fast Charger (Blue)**
    - **AC Level 1 (Orange)**
    - **AC Level 2 (Green)**
    
    **DC Charging tends to be the most expensive** given it is the fastest and so required **the largest grid connection and most expensive cabling and connection equipment**

    We can see there is a slight variance between city costs. This is because major deregulated U.S. electricity markets,
                 use nodal pricing (Locational Marginal Pricing or LMP) 
                to set different prices at thousands of grid "nodes" based on local supply, demand, and transmission congestion, reflecting real-time costs.
    """)
    # Color scheme (same as all other charts)
    color_map = {
        "Level 1": "orange",
        "Level 2": "green",
        "DC Fast Charger": "lightblue"
    }

    # Filter by selected charger types
    if select_charger_types_sidebar:
        df_city_cost = df[df["Charger Type"].isin(select_charger_types_sidebar)].copy()
    else:
        df_city_cost = df.copy()

    # Ensure numeric
    df_city_cost["Cost Per kWh ($)"] = pd.to_numeric(df_city_cost["Cost Per kWh ($)"], errors="coerce")

    # Group by city and charger type
    city_avg_cost = df_city_cost.groupby(
        ["Charging Station Location", "Charger Type"], 
        as_index=False
    )["Cost Per kWh ($)"].mean()

    city_avg_cost["Cost Per kWh ($)"] = city_avg_cost["Cost Per kWh ($)"].round(2)

    # Build grouped bar chart
    fig_city_bar = px.bar(
        city_avg_cost,
        x="Charging Station Location",
        y="Cost Per kWh ($)",
        color="Charger Type",
        barmode="group",
        text="Cost Per kWh ($)",
        title="Average Cost per kWh ($) by City and Charger Type",
        color_discrete_map=color_map
    )

    fig_city_bar.update_layout(
        xaxis_title="City",
        yaxis_title="Average Cost ($/kWh)",
        height=500
    )

    st.plotly_chart(fig_city_bar, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("""
    The scatter plot below shows the **average cost per unit ($/kWh)** by **charger type** over the course of a typical day.
    """)
    st.info("Select your city and charger type to see the most cost effective time of day to charge your wagon!")
    selected_cities = df_filtered["Charging Station Location"].unique()
    df_tab2 = df[df["Charging Station Location"].isin(selected_cities)].copy()

    if df_tab2.empty:
        st.info("No data available to plot Cost per kWh by time of day.")
    else:
        # Ensure necessary columns are numeric/datetime
        df_tab2["Charging Start Time"] = pd.to_datetime(df_tab2["Charging Start Time"], errors="coerce")
        df_tab2["Cost Per kWh ($)"] = pd.to_numeric(df_tab2["Cost Per kWh ($)"], errors="coerce")

        # Filter by selected charger types
        df_plot = df_tab2[df_tab2["Charger Type"].isin(select_charger_types_sidebar)]

        # Extract time of day as decimal hours
        df_plot["Hour of Day"] = df_plot["Charging Start Time"].dt.hour + df_plot["Charging Start Time"].dt.minute / 60

        # Keep only valid hours (0-24)
        df_plot = df_plot[df_plot["Hour of Day"].between(0, 24)]

    if df_plot.empty:
        st.info("No valid data after filtering by charger type and time of day.")
    else:
        # Define custom colors
        color_map = {
            "Level 1": "orange",
            "Level 2": "green",
            "DC Fast Charger": "lightblue"
        }

        # Build title including filtered city/cities
        if len(selected_cities) == 1:
            plot_title = f"{selected_cities[0]} – Cost per Unit ($/kWh) by Charger Type"
        else:
            plot_title = f"Cost per Unit ($/kWh) by Charger Type – Multiple Cities"

        # Create scatter plot with trend lines
        fig_scatter = px.scatter(
            df_plot,
            x="Hour of Day",
            y="Cost Per kWh ($)",
            color="Charger Type",
            color_discrete_map=color_map,
            hover_data=["Charging Station Location", "User ID", "Charging Start Time"],
            trendline="ols",  # Add trend line
            trendline_scope="group",  # Separate trend line per charger type
            opacity=0.5,
            title=plot_title
        )

        # Make markers slightly larger for clarity
        fig_scatter.update_traces(marker=dict(size=25), selector=dict(mode='markers'))

        # Set x-axis to strictly 0-24 hours and y-axis title
        fig_scatter.update_layout(
            xaxis=dict(title="Hour of Day", range=[0, 24], tickmode="linear", dtick=1),
            yaxis=dict(title="Cost Per kWh ($)"),
            title=dict(x=0.25)  # Center the title
        )

        # Display the plot
        st.plotly_chart(fig_scatter, use_container_width=True)

    if df_tab2.empty:
        st.info("No data available for the selected city/cities.")
    else:
        # Aggregate: average cost per charger type and total kWh consumed
        summary_table = df_tab2.groupby("Charger Type", as_index=False).agg(
            **{
                "Average Cost per kWh ($)": ("Cost Per kWh ($)", "mean"),
                "Total kWh Used (kWh)": ("Energy Consumed (kWh)", "sum")
            }
        )

        # round numeric values for readability
        summary_table["Average Cost per kWh ($)"] = summary_table["Average Cost per kWh ($)"].round(2)
        summary_table["Total kWh Used (kWh)"] = summary_table["Total kWh Used (kWh)"].round(0)

        # Display in an expander without index
        with st.expander("Average Cost ($/kWh) By Charger Type"):
            st.dataframe(summary_table)  # as_index=False ensures no index is added
    
    if df_tab2.empty:
        st.info("No data available for the selected city/cities.")
    else:
        # Aggregate: average cost and total kWh by Charging Station ID and Charger Type
        summary_table = df_tab2.groupby(["Charging Station ID", "Charger Type"], as_index=False).agg(
            **{
                "Average Cost per kWh ($)": ("Cost Per kWh ($)", "mean"),
                "Total kWh Used (kWh)": ("Energy Consumed (kWh)", "sum")
            }
        )

        # round numeric values for readability
        summary_table["Average Cost per kWh ($)"] = summary_table["Average Cost per kWh ($)"].round(2)
        summary_table["Total kWh Used (kWh)"] = summary_table["Total kWh Used (kWh)"].round(0)

        # Display in an expander
        with st.expander("Average Cost ($/kWh) by Station ID"):
            st.dataframe(summary_table)
    
    st.info("Select your city and preferred charge point type to view the cheapest charge points!" \
    " As you can see there is a large variance between costs, so please ensure you choose the right one")
    # Sort by average cost ascending (cheapest first)
    summary_table = summary_table.sort_values("Average Cost per kWh ($)")

    if df_tab2.empty:
        st.info("No data available for the selected city/cities.")
    else:
        # Filter by selected charger types
        if select_charger_types_sidebar:
            df_filtered_chart = df_tab2[df_tab2["Charger Type"].isin(select_charger_types_sidebar)].copy()
        else:
            df_filtered_chart = df_tab2.copy()

        # Aggregate: average cost and total kWh by Charging Station ID and Charger Type
        summary_table = df_filtered_chart.groupby(
            ["Charging Station ID", "Charger Type"], as_index=False
        ).agg(
            **{
                "Average Cost per kWh ($)": ("Cost Per kWh ($)", "mean"),
                "Total kWh Used (kWh)": ("Energy Consumed (kWh)", "sum")
            }
        )

        # Round numeric values
        summary_table["Average Cost per kWh ($)"] = summary_table["Average Cost per kWh ($)"].round(2)
        summary_table["Total kWh Used (kWh)"] = summary_table["Total kWh Used (kWh)"].round(0)

        # Sort cheapest to most expensive
        summary_table = summary_table.sort_values("Average Cost per kWh ($)")

        # Horizontal bar chart
        fig_bar = px.bar(
            summary_table,
            y="Charging Station ID",
            x="Average Cost per kWh ($)",
            color="Charger Type",
            orientation="h",
            text="Average Cost per kWh ($)",
            color_discrete_map={
                "Level 1": "orange",
                "Level 2": "green",
                "DC Fast Charger": "lightblue"
            },
            title="Average Cost per kWh ($/kWh) by Charge Point"
        )

        fig_bar.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Average Cost per kWh ($)",
            yaxis_title="Charging Station ID",
            legend_title="Charger Type",
            height=1000  # double the default height for more visibility
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    col1, col2  = st.columns(2)

    with col1:
        st.write("##### Expenditure by Charge Type")
        # Filter main df using selected cities

        if df_tab2.empty:
            st.info("No matching data for selected filters.")
        else:
            # Ensure Charging Cost is numeric
            df_tab2["Charging Cost (USD)"] = pd.to_numeric(df_tab2["Charging Cost (USD)"], errors="coerce")
            
            # Aggregate total cost per Charger Type
            expenditure = df_tab2.groupby("Charger Type")["Charging Cost (USD)"].sum().reset_index()

            # Define chart title
            if len(selected_cities) == 1:
                chart_title = f"{selected_cities[0]} – Expenditure by Charger Type"
            else:
                chart_title = "Multiple Cities – Expenditure by Charger Type"

            # Define custom colors
            color_map = {
                "Level 1": "orange",        # Sun orange
                "Level 2": "green",         # Forest green
                "DC Fast Charger": "lightblue"  # Light blue
            }

            # Plot pie chart
            fig = px.pie(
                expenditure,
                names="Charger Type",
                values="Charging Cost (USD)",
                title=chart_title,
                hole=0.3,
                color="Charger Type",
                color_discrete_map=color_map
            )

            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### This pie chart shows the distribution of customer spending across different charger types.")
     
    with col2:
        st.markdown("##### Total Number of Installations")

        if df_tab2.empty:
            st.info("No data available to plot charger installation distribution.")
        else:
            # Count number of chargers by type within each city
            charger_counts = df_tab2.groupby(
                ["Charging Station Location", "Charger Type"]
            ).size().reset_index(name="Number of Chargers")

            # Define chart title
            if len(selected_cities) == 1:
                chart_title2 = f"{selected_cities[0]} – Number of Charge Points Installed"
            else:
                chart_title2 = "Multiple Cities – Number of Charge Points Installed"

            # Define custom colors
            color_map = {
                "Level 1": "orange",        # Sun orange
                "Level 2": "green",         # Forest green
                "DC Fast Charger": "lightblue"  # Light blue
            }

            # Plot grouped bar chart
            fig_bar = px.bar(
                charger_counts,
                x="Charging Station Location",
                y="Number of Chargers",
                color="Charger Type",
                barmode="group",  # use "stack" if you want stacked bars
                color_discrete_map=color_map,
                title=chart_title2,
                text="Number of Chargers"
            )

            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown(
                "#### This bar chart shows how many charging stations of each type are installed"
            )
#=============================================================================================
# Tab 2 END

# Tab 3 START
#=============================================================================================
    with tab3:
        st.header("Charge Point Avaialbilty and Utilisation Stats")
        
        col1, col2  = st.columns(2)
        
        with col1:
            st.markdown("""The intereactive plotly bar chart below shows the **number of charging sessions per hour** in your city.
                        """)
            st.info("Select your city and preferred charge types to view the most likely time a charger will be available to you")
            # Plot Plotly Stacked Bar chart showing number of sessions per hour in given city
            # Apply sidebar filters to df_usage
            df_plot = df_usage[
                (df_usage["Charging Station Location"] == select_city_sidebar) &
                (df_usage["Charger Type"].isin(select_charger_types_sidebar))
            ].copy()

            if df_plot.empty:
                st.info("No data available for the selected city and charger types.")
            else:
                # Convert to datetime
                df_plot["Charging Start Time"] = pd.to_datetime(
                    df_plot["Charging Start Time"],
                    errors="coerce"
                )

                # Extract hour of day only
                df_plot["Hour of Day"] = df_plot["Charging Start Time"].dt.hour

                # Drop invalid rows
                df_plot = df_plot.dropna(subset=["Hour of Day", "Charger Type"])

                # Count sessions by hour + charger type
                hourly_counts = (
                    df_plot
                    .groupby(["Hour of Day", "Charger Type"])
                    .size()
                    .reset_index(name="Number of Sessions")
                )

                # Custom colors
                color_map = {
                    "Level 1": "#FFA500",         # Sun orange
                    "Level 2": "#007700",         # Dark green
                    "DC Fast Charger": "#1E90FF"  # Blue
                }

                # Dynamic title
                plot_title = f"{select_city_sidebar} – Number of Charge Sessions by Hour"

            with col2:
                # Plotly stacked bar chart
                fig = px.bar(
                    hourly_counts,
                    x="Hour of Day",
                    y="Number of Sessions",
                    color="Charger Type",
                    color_discrete_map=color_map,
                    title=plot_title
                )

                fig.update_layout(
                    barmode="stack",
                    xaxis=dict(dtick=1)
                )

                st.plotly_chart(fig, use_container_width=True)
            
                st.markdown("""The intereactive plotly bar chart above shows the **number of charging sessions per day of the week** in your city.
                            """)
                st.info("Select your city and preferred charge types to view the most likely time a charger will be available to you")
            # Apply sidebar filters to df_usage
            df_plot = df_usage[
                (df_usage["Charging Station Location"] == select_city_sidebar) &
                (df_usage["Charger Type"].isin(select_charger_types_sidebar))
            ].copy()

            if df_plot.empty:
                st.info("No data available for the selected city and charger types.")
            else:
                # Ensure Day of Week is consistent
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                df_plot["Day of Week"] = pd.Categorical(df_plot["Day of Week"], categories=day_order, ordered=True)

                # Count sessions per day and charger type
                daily_counts = (
                    df_plot
                    .groupby(["Day of Week", "Charger Type"])["Charging Station ID"]
                    .count()
                    .reset_index(name="Number of Sessions")
                )

                # Custom color map
                color_map = {
                    "Level 1": "#FFA500",         # Sun orange
                    "Level 2": "#007700",         # Dark green
                    "DC Fast Charger": "#1E90FF"  # Blue
                }

                # Dynamic title
                plot_title = f"{select_city_sidebar} – Number of Charge Sessions by Day of the Week"

                # Plotly grouped bar chart
                fig = px.bar(
                    daily_counts,
                    x="Day of Week",
                    y="Number of Sessions",
                    color="Charger Type",
                    color_discrete_map=color_map,
                    barmode="group",
                    category_orders={"Day of Week": day_order},
                    title=plot_title
                )

                fig.update_layout(
                    xaxis_title="Day of Week",
                    yaxis_title="Number of Sessions",
                    xaxis_tickangle=-45
                )

                st.plotly_chart(fig, use_container_width=True)
            
        # Apply sidebar filters
        df_plot = df_usage[
            (df_usage["Charging Station Location"] == select_city_sidebar) &
            (df_usage["Charger Type"].isin(select_charger_types_sidebar))
        ].copy()

        if df_plot.empty:
            st.info("No data available for the selected city and charger types.")
        else:
            # Ensure datetime and extract hour
            df_plot["Charging Start Time"] = pd.to_datetime(
                df_plot["Charging Start Time"], errors="coerce"
            )
            df_plot["Hour"] = df_plot["Charging Start Time"].dt.hour
            df_plot = df_plot.dropna(subset=["Hour", "Charger Type"])

            # Group by hour and charger type
            hourly_counts = (
                df_plot.groupby(["Hour", "Charger Type"])
                .size()
                .reset_index(name="Number of Sessions")
            )

            # Custom colour mapping
            color_map = {
                "Level 1": "#FFA500",         # Sun orange
                "Level 2": "#007700",         # Forest green
                "DC Fast Charger": "#1E90FF"  # Blue
            }

            # Dynamic title
            plot_title = f"Peak Hours by Charger Type – {select_city_sidebar}"

            # Plotly line chart
            fig = px.line(
                hourly_counts,
                x="Hour",
                y="Number of Sessions",
                color="Charger Type",
                color_discrete_map=color_map,
                markers=True,
                title=plot_title,
                labels={
                    "Hour": "Hour of Day",
                    "Number of Sessions": "Number of Charging Sessions"
                }
            )

            # Set x-axis strictly 0–23
            fig.update_layout(
                xaxis=dict(range=[0, 23], dtick=1),
                yaxis=dict(title="Number of Sessions"),
                legend_title_text="Charger Type"
            )

            st.plotly_chart(fig, use_container_width=True)
#=============================================================================================
# Tab 3 END

# Tab 4 START
#=============================================================================================
    with tab4:
        st.header("EV Car Market Analysis")
            # Show dataframe inside an expander
        unique_models = df_market["model"].nunique()
        st.markdown(f"""
                    In this section we analyse **{unique_models}** car models to help you buy the right model with specialised insights across:
                    - Efficiency
                    - Cost
                    - Range
                    - Charging capabilities

                    The dataframe below is the data we evaluate in this app and can be downloaded onto your computer for further analysis
                    """)

        with st.expander("View EV Market Data"):
            st.dataframe(df_market, use_container_width=True)
        
        col1, col2 = st.columns(2)
        # Define custom colors
        COLOUR_SCHEME_BODY = {
            "SUV": "#007700",       # Forest green
            "Sedan": "#FFA500",     # Sun orange
            "Hatchback": "#0D6EFD"  # Blue
        }

        COLOUR_SCHEME_CATEGORY = {
            "Economy": "#007700",   # Forest green
            "Mid-Range": "#FFA500",       # Sun orange
            "Premium": "#0D6EFD"   # Blue
        }
        with col1:
            st.markdown("""
            The below box plot shows the efficiency of cars by body type (km/kWh - distance per unit of energy)
            and shows **hatchback cars** clearly have the **highest efficiency per kilometre** and are therefore the cheapest to run
            """)

            #Box plot body type vs. efficiency per km
            # Convert efficiency (Wh/km -> km/kWh)
            df_market["Efficiency_km_kWh"] = 1000 / df_market["efficiency_wh_per_km"]

            # Filter to main types
            main_types = df_market[df_market["Body_Type"].isin(["SUV", "Sedan", "Hatchback"])]

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                data=main_types,
                x="Body_Type",
                y="Efficiency_km_kWh",
                hue="Body_Type",
                palette=COLOUR_SCHEME_BODY,
                legend=False,
                ax=ax
            )

            ax.set_title("Real-World Efficiency by Car Body Type")
            ax.set_ylabel("Efficiency (km per kWh)")

            # Display in Streamlit
            st.pyplot(fig)

        with col2:
            # Scatterplot showing price vs. efficiency
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(
                data=df_market,
                x='Efficiency_km_kWh',
                y='Estimated_Price',
                hue='Category',       # Color by segment
                style='Body_Type',    # Different marker for body type
                palette=COLOUR_SCHEME_CATEGORY,
                s=100,
                alpha=0.8,
                ax=ax
            )

            # Title and labels
            ax.set_title("Market Efficiency: Price vs. Efficiency (by Segment & Type)")
            ax.set_xlabel("Efficiency (km per kWh)")
            ax.set_ylabel("Estimated Price ($)")

            # Place legend outside the plot
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

            # Display in Streamlit
            st.pyplot(fig)
            
            st.markdown("""
            The above scatter plot shows the efficiency of cars vs. cost (km/kWh - distance per unit of energy)
            interestingly we can see the low cost cars also tend to have the **highest efficiency per kilometre**
            """)

        # Scatterplot showing charging capability vs. price
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            data=df_market,
            x='Estimated_Price',
            y='fast_charging_power_kw_dc',
            hue='Category',
            palette=COLOUR_SCHEME_CATEGORY,
            s=100,
            ax=ax
        )

        # Titles and labels
        ax.set_title("Charging Capability: Price vs. Max DC Charging Speed")
        ax.set_xlabel("Estimated Car Price ($)")
        ax.set_ylabel("Max DC Charging Speed (kW)")

        # Add threshold line
        ax.axhline(
            y=150,
            color='r',
            linestyle='--',
            label="High Speed Threshold (150 kW)"
        )

        # Place legend
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

        # Tight layout for better spacing
        plt.tight_layout()

        # Display plot in Streamlit
        st.pyplot(fig)

        # PREDICTOR: CHEAPEST MILE PER CITY - USING BOTH DFs
        city_costs = df_usage.groupby('Charging Station Location')['Price per kWh'].mean().reset_index()

        # Define comparison cars
        comparison_cars = [
            {'Model': 'Nissan Leaf (Economy)', 'Eff_km_kWh': 5.5}, 
            {'Model': 'Tesla Model 3 (Mid)',   'Eff_km_kWh': 6.5}, 
            {'Model': 'BMW i3 (Premium)',      'Eff_km_kWh': 6.0}, 
            {'Model': 'Hummer EV (Premium)',   'Eff_km_kWh': 2.5}
        ]
        df_compare = pd.DataFrame(comparison_cars)
        
        COLOUR_SCHEME_CATEGORY = {
            "Nissan Leaf (Economy)": "#007700",   # Forest green
            "Tesla Model 3 (Mid)": "#FFA500",     # Sun orange
            "BMW i3 (Premium)": "#0D6EFD",        # Blue
            "Hummer EV (Premium)": "#FFB6C1"      # Light pink
        }

        # Merge city costs with car efficiency
        df_pred = pd.merge(
            city_costs.assign(key=1),
            df_compare.assign(key=1),
            on='key'
        ).drop('key', axis=1)

        # Calculate predicted cost per km
        df_pred['Cost_Per_Km'] = df_pred['Price per kWh'] / df_pred['Eff_km_kWh']
        
        st.info("The below bar chart shows the cost of running selected vehicles ($/km) depending which city they charge in")
        
        col1, col2 = st.columns(2)

        with col1:
            selected_cities = st.multiselect(
                "Select Cities",
                options=sorted(df_pred["Charging Station Location"].unique()),
                default=sorted(df_pred["Charging Station Location"].unique())
            )

        with col2:
            selected_models = st.multiselect(
                "Select Car Models",
                options=sorted(df_pred["Model"].unique()),
                default=sorted(df_pred["Model"].unique())
            )

        # Apply filters
        df_pred_filtered = df_pred[
            df_pred["Charging Station Location"].isin(selected_cities) &
            df_pred["Model"].isin(selected_models)
        ]

        # Create color sequence in the order of filtered models
        color_sequence = [COLOUR_SCHEME_CATEGORY[m] for m in df_pred_filtered["Model"].unique()]

        # Plot interactive bar chart
        if not df_pred_filtered.empty:
            fig_pred = px.bar(
                df_pred_filtered,
                x="Charging Station Location",
                y="Cost_Per_Km",
                color="Model",
                text="Cost_Per_Km",
                barmode="group",
                color_discrete_sequence=color_sequence,  # use list of colors
                labels={"Cost_Per_Km": "Cost ($) per Km", "Charging Station Location": "City"},
                title="Cost ($/per Km) by City and Car Model"
            )
            fig_pred.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_pred.update_layout(
                xaxis_tickangle=-45,
                yaxis_title="Cost ($) per Km",
                height=600,
                legend_title="Car Model"
            )

            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info("No data available for the selected city or car model filters.")
#=============================================================================================
# Tab 4 END

# Footer
st.markdown("---")
st.markdown("**©** Copyright protected by The EV Plot est. 2026 (time-travel is the surest way to be ahead of your time)")

# Row container with quote on the right
def get_gif_base64(file_path):
    with open(file_path, "rb") as f:
        content = f.read()
        return base64.b64encode(content).decode("utf-8")

# Path to your local GIF
gif_path = 'charging_gif3.gif'
gif_data = get_gif_base64(gif_path)

st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; margin-top: 20px;">
        <img src="data:image/gif;base64,{gif_data}" alt="Animated GIF" width="300">
        <div style="color: #007700; font-size: 32px; font-weight: bold; margin-left: 20px; text-align: right;">
            "The only bad thing about EVs (aside from Elon Musk)<br>is finding the cheapest, nearby charging station"<br>– Pengu
        </div>
    </div>
    """,
    unsafe_allow_html=True
)