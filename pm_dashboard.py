import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap

# Set page configuration
st.set_page_config(
    page_title="Italian Air Quality Dashboard",
    page_icon="🇮🇹",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3F66;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E5984;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Dashboard title
st.markdown('<p class="main-header">Italian Municipalities Air Quality & Emissions Dashboard (2011-2022)</p>', unsafe_allow_html=True)

# Display app information
with st.expander("About This Dashboard"):
    st.markdown("""
    This dashboard visualizes PM10, PM2.5, and CO2 emissions data for Italian municipalities from 2011 to 2022.
    - Upload your CSV files using the sidebar uploader
    - Filter by municipality, year range, and pollutant type
    - Explore different visualizations including time series, comparisons, maps, and statistical summaries
    
    If you encounter any issues with your data format, check the debug information in the sidebar.
    """)

def create_sample_data_with_emissions_format(pollutant_type):
    """Create sample data for demonstration using the emissions_YYYY format"""
    municipalities = ["Milan", "Rome", "Naples", "Turin", "Florence", 
                      "Bologna", "Venice", "Genoa", "Palermo", "Bari"]
    
    # Sample coordinates for the municipalities (for mapping)
    coordinates = {
        "Milan": [45.4642, 9.1900],
        "Rome": [41.9028, 12.4964],
        "Naples": [40.8518, 14.2681],
        "Turin": [45.0703, 7.6869],
        "Florence": [43.7696, 11.2558],
        "Bologna": [44.4949, 11.3426],
        "Venice": [45.4408, 12.3155],
        "Genoa": [44.4056, 8.9463],
        "Palermo": [38.1157, 13.3615],
        "Bari": [41.1171, 16.8719]
    }
    
    # Create emissions_YYYY columns (2011 to 2022)
    data = {}
    for year in range(2011, 2023):
        # Different base values depending on pollutant type
        if pollutant_type == "PM10":
            base_value = 40
            multiplier = 1.5
            variation = 10
        elif pollutant_type == "PM2.5":
            base_value = 25
            multiplier = 1.0
            variation = 8
        else:  # CO2
            base_value = 8000  # Higher value for CO2 (in tons)
            multiplier = 100
            variation = 2000
            
        # Generate random data with a decreasing trend
        data[f'emissions_{year}'] = np.random.normal(
            base_value - (year - 2011) * multiplier, 
            variation, 
            size=len(municipalities)
        )
    
    # Add latitude and longitude columns for mapping
    data['lat'] = [coordinates[city][0] for city in municipalities]
    data['lon'] = [coordinates[city][1] for city in municipalities]
    
    # Create DataFrame
    df = pd.DataFrame(data, index=municipalities)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'COMUNE'}, inplace=True) # Use 'COMUNE' to match your data format
    
    return df

# Function to detect and prepare data for time series plots
def reshape_for_timeseries(df, pollutant_type):
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Handle specific column format for the provided data
    # Identify emissions columns (format 'emissions_YYYY')
    year_columns = [col for col in df_copy.columns if col.startswith('emissions_')]
    
    # Debug information
    st.sidebar.markdown("### Debug Info")
    with st.sidebar.expander("Column Detection"):
        st.write(f"Detected year columns: {year_columns}")
    
    # If no emissions columns found, try generic approach
    if not year_columns:
        # Try other common patterns
        year_columns = [col for col in df_copy.columns if col.isdigit() and 2000 <= int(col) <= 2030]
        
        if not year_columns:
            # Last resort: try to guess based on numeric columns, excluding municipality column
            municipality_col = 'COMUNE' if 'COMUNE' in df_copy.columns else 'Municipality'
            year_columns = [col for col in df_copy.columns if col != municipality_col and pd.api.types.is_numeric_dtype(df_copy[col]) 
                           and col not in ['lat', 'lon']]  # Exclude lat/lon columns
    
    # If still no columns found, return empty dataframe with correct structure
    if not year_columns:
        st.error(f"Could not identify year columns for {pollutant_type} data")
        return pd.DataFrame(columns=['Municipality', 'Year', 'Value'])
    
    # Create melted dataframe for time series
    try:
        # Determine the municipality column name
        municipality_col = 'COMUNE' if 'COMUNE' in df_copy.columns else 'Municipality'
        
        df_melted = df_copy.melt(
            id_vars=[municipality_col],
            value_vars=year_columns,
            var_name='Year',
            value_name='Value'
        )
        
        # Extract year from 'emissions_YYYY' format
        if len(df_melted) > 0 and 'emissions_' in df_melted['Year'].iloc[0]:
            df_melted['Year'] = df_melted['Year'].str.replace('emissions_', '').astype(int)
        elif len(df_melted) > 0 and '_' in df_melted['Year'].iloc[0]:
            # Handle other formats with underscore
            df_melted['Year'] = df_melted['Year'].str.split('_').str[1].astype(int)
        else:
            # Assume direct year format like '2011'
            df_melted['Year'] = df_melted['Year'].astype(int)
        
        # Rename municipality column to standardize
        df_melted.rename(columns={municipality_col: 'Municipality'}, inplace=True)
        
        return df_melted
    except Exception as e:
        st.error(f"Error reshaping data: {e}")
        # Print the detailed error for debugging
        st.code(str(e))
        return pd.DataFrame(columns=['Municipality', 'Year', 'Value'])

# Get Italian GeoJSON for mapping
@st.cache_data
def load_data():
    try:
        # Add file uploader for data files
        st.sidebar.markdown("### Upload Data Files")
        pm10_file = st.sidebar.file_uploader("Upload PM10 CSV file", type=['csv'])
        pm25_file = st.sidebar.file_uploader("Upload PM2.5 CSV file", type=['csv'])
        co2_file = st.sidebar.file_uploader("Upload CO2 CSV file", type=['csv'])
        
        if pm10_file is not None:
            df_pm10 = pd.read_csv(pm10_file)
            st.sidebar.success("PM10 data loaded successfully!")
        else:
            # Use sample data for demonstration with the expected column format
            st.sidebar.warning("Using sample PM10 data for demonstration")
            df_pm10 = create_sample_data_with_emissions_format("PM10")
            
        if pm25_file is not None:
            df_pm25 = pd.read_csv(pm25_file)
            st.sidebar.success("PM2.5 data loaded successfully!")
        else:
            # Use sample data for demonstration with the expected column format
            st.sidebar.warning("Using sample PM2.5 data for demonstration")
            df_pm25 = create_sample_data_with_emissions_format("PM2.5")
            
        if co2_file is not None:
            df_co2 = pd.read_csv(co2_file)
            st.sidebar.success("CO2 data loaded successfully!")
        else:
            # Use sample data for demonstration with the expected column format
            st.sidebar.warning("Using sample CO2 data for demonstration")
            df_co2 = create_sample_data_with_emissions_format("CO2")
        
        # Display data structure information
        with st.sidebar.expander("View Data Structure"):
            st.markdown("#### PM10 DataFrame Columns")
            st.write(df_pm10.columns.tolist())
            st.markdown("#### PM2.5 DataFrame Columns")
            st.write(df_pm25.columns.tolist())
            st.markdown("#### CO2 DataFrame Columns")
            st.write(df_co2.columns.tolist())
        
        return df_pm10, df_pm25, df_co2
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Fallback to sample data with the expected column format
        return create_sample_data_with_emissions_format("PM10"), create_sample_data_with_emissions_format("PM2.5"), create_sample_data_with_emissions_format("CO2")

# Try to load data from files first
try:
    # Check if data files exist
    pm10_file_path = 'Data/total_PM10_2011_2022_whole_italy_emissions.csv'
    pm25_file_path = 'Data/total_PM2.5_2011_2022_whole_italy_emissions.csv'
    co2_file_path = 'Data/total_CO2_2011_2022_whole_italy_emissions.csv'
    
    if os.path.exists(pm10_file_path) and os.path.exists(pm25_file_path):
        df_pm10 = pd.read_csv(pm10_file_path)
        df_pm25 = pd.read_csv(pm25_file_path)
        
        # Check for CO2 data, create sample if not available
        if os.path.exists(co2_file_path):
            df_co2 = pd.read_csv(co2_file_path)
        else:
            st.warning("CO2 data file not found. Using sample CO2 data.")
            df_co2 = create_sample_data_with_emissions_format("CO2")
            
            # Copy municipality and coordinate information from PM10 data if available
            if 'COMUNE' in df_pm10.columns:
                muni_col = 'COMUNE'
                # Create a new dataframe with the same municipalities
                base_data = df_pm10[[muni_col]].copy()
                
                # Add coordinates if available
                if 'lat' in df_pm10.columns and 'lon' in df_pm10.columns:
                    base_data['lat'] = df_pm10['lat']
                    base_data['lon'] = df_pm10['lon']
                
                # Add sample CO2 emissions data
                for year in range(2011, 2023):
                    base_data[f'emissions_{year}'] = np.random.normal(
                        8000 - (year - 2011) * 100, 
                        2000, 
                        size=len(base_data)
                    )
                
                df_co2 = base_data
    else:
        # Fall back to the data loading function if files don't exist
        df_pm10, df_pm25, df_co2 = load_data()
except Exception as e:
    st.error(f"Error loading data files: {e}")
    # Fall back to data loading function
    df_pm10, df_pm25, df_co2 = load_data()

# Load GeoJSON for Italy (if available)
italy_geojson = load_italy_geojson()

# Sidebar for filtering
st.sidebar.markdown('## Filters')

# Detect min and max years from data based on emissions_YYYY format
pm10_year_cols = [col for col in df_pm10.columns if col.startswith('emissions_')]
pm25_year_cols = [col for col in df_pm25.columns if col.startswith('emissions_')]
co2_year_cols = [col for col in df_co2.columns if col.startswith('emissions_')]

# Extract years from emissions_YYYY format
all_years = []

def extract_years(year_cols):
    years = []
    if year_cols:
        years = [int(col.replace('emissions_', '')) for col in year_cols]
    return years

pm10_years = extract_years(pm10_year_cols)
pm25_years = extract_years(pm25_year_cols)
co2_years = extract_years(co2_year_cols)

# Combine all detected years
all_years = pm10_years + pm25_years + co2_years

# Set min and max years for the slider
min_year = min(all_years) if all_years else 2011
max_year = max(all_years) if all_years else 2022

# Municipality selection
municipality_column = 'COMUNE' if 'COMUNE' in df_pm10.columns else 'Municipality'
if municipality_column in df_pm10.columns:
    # Convert all values to strings before sorting to avoid type comparison issues
    municipalities = df_pm10[municipality_column].astype(str).unique()
    # Now sort the string values
    municipalities = sorted(municipalities)
    
    default_municipalities = municipalities[:min(5, len(municipalities))]
    
    selected_municipalities = st.sidebar.multiselect(
        'Select Municipalities',
        options=municipalities,
        default=default_municipalities
    )
else:
    st.error(f"Column '{municipality_column}' not found in the data. Please ensure your CSV has a municipality column ('COMUNE' or 'Municipality').")
    selected_municipalities = []

# Year range selection
year_range = st.sidebar.slider(
    'Select Year Range',
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Single year selection for maps
map_year = st.sidebar.slider(
    'Select Year for Maps',
    min_value=min_year,
    max_value=max_year,
    value=max_year
)

# Pollutant type selection
pollutant_type = st.sidebar.radio(
    'Select Pollutant Type',
    options=['PM10', 'PM2.5', 'CO2', 'All']
)

# Main dashboard content
if not selected_municipalities:
    st.warning("Please select at least one municipality from the sidebar to display visualizations.")
else:
    # Tab layout for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Comparisons", "Maps", "Statistics"])
    
    # Time Series Tab
    with tab1:
        st.markdown('<p class="sub-header">Time Series Analysis</p>', unsafe_allow_html=True)
        
        try:
            # Filter data based on selections
            if pollutant_type in ['PM10', 'All']:
                # Prepare PM10 data
                pm10_timeseries = reshape_for_timeseries(df_pm10, 'PM10')
                
                if not pm10_timeseries.empty:
                    pm10_filtered = pm10_timeseries[
                        (pm10_timeseries['Municipality'].isin(selected_municipalities)) &
                        (pm10_timeseries['Year'] >= year_range[0]) &
                        (pm10_timeseries['Year'] <= year_range[1])
                    ]
                    
                    if not pm10_filtered.empty:
                        # Create PM10 time series plot
                        fig_pm10 = px.line(
                            pm10_filtered,
                            x='Year',
                            y='Value',
                            color='Municipality',
                            title='PM10 Levels Over Time',
                            labels={'Value': 'PM10 (μg/m³)'}
                        )
                        fig_pm10.update_layout(
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_pm10, use_container_width=True)
                    else:
                        st.info("No PM10 data available for the selected filters.")
                else:
                    st.info("Could not process PM10 data. Please check your data format.")
            
            if pollutant_type in ['PM2.5', 'All']:
                # Prepare PM2.5 data
                pm25_timeseries = reshape_for_timeseries(df_pm25, 'PM2.5')
                
                if not pm25_timeseries.empty:
                    pm25_filtered = pm25_timeseries[
                        (pm25_timeseries['Municipality'].isin(selected_municipalities)) &
                        (pm25_timeseries['Year'] >= year_range[0]) &
                        (pm25_timeseries['Year'] <= year_range[1])
                    ]
                    
                    if not pm25_filtered.empty:
                        # Create PM2.5 time series plot
                        fig_pm25 = px.line(
                            pm25_filtered,
                            x='Year',
                            y='Value',
                            color='Municipality',
                            title='PM2.5 Levels Over Time',
                            labels={'Value': 'PM2.5 (μg/m³)'}
                        )
                        fig_pm25.update_layout(
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_pm25, use_container_width=True)
                    else:
                        st.info("No PM2.5 data available for the selected filters.")
                else:
                    st.info("Could not process PM2.5 data. Please check your data format.")
                    
            if pollutant_type in ['CO2', 'All']:
                # Prepare CO2 data
                co2_timeseries = reshape_for_timeseries(df_co2, 'CO2')
                
                if not co2_timeseries.empty:
                    co2_filtered = co2_timeseries[
                        (co2_timeseries['Municipality'].isin(selected_municipalities)) &
                        (co2_timeseries['Year'] >= year_range[0]) &
                        (co2_timeseries['Year'] <= year_range[1])
                    ]
                    
                    if not co2_filtered.empty:
                        # Create CO2 time series plot
                        fig_co2 = px.line(
                            co2_filtered,
                            x='Year',
                            y='Value',
                            color='Municipality',
                            title='CO2 Emissions Over Time',
                            labels={'Value': 'CO2 (tonnes)'}
                        )
                        fig_co2.update_layout(
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_co2, use_container_width=True)
                    else:
                        st.info("No CO2 data available for the selected filters.")
                else:
                    st.info("Could not process CO2 data. Please check your data format.")
        except Exception as e:
            st.error(f"Error creating time series plots: {e}")
    
    # Comparisons Tab    
    with tab2:
        st.markdown('<p class="sub-header">Municipality Comparison</p>', unsafe_allow_html=True)
        
        try:
            # Select most recent year in the range
            latest_year = min(max_year, year_range[1])
            
            # Determine municipality column name
            muni_col = 'COMUNE' if 'COMUNE' in df_pm10.columns else 'Municipality'
            
            # Create layout with columns
            if pollutant_type == 'All':
                col1, col2 = st.columns(2)
                
                with col1:
                    # PM10 and PM2.5 Comparisons
                    create_comparison_bar_chart(df_pm10, muni_col, selected_municipalities, latest_year, "PM10", "Reds")
                    create_comparison_bar_chart(df_pm25, muni_col, selected_municipalities, latest_year, "PM2.5", "Oranges")
                
                with col2:
                    # CO2 Comparison
                    create_comparison_bar_chart(df_co2, muni_col, selected_municipalities, latest_year, "CO2", "Greens", "tonnes")
                    
                    # Add a trend analysis
                    st.markdown("### Trend Analysis")
                    create_trend_chart(df_pm10, df_pm25, df_co2, muni_col, selected_municipalities, year_range)
                
            else:
                # Based on selected pollutant
                if pollutant_type == "PM10":
                    df = df_pm10
                    color_scale = "Reds"
                    units = "μg/m³"
                elif pollutant_type == "PM2.5":
                    df = df_pm25
                    color_scale = "Oranges"
                    units = "μg/m³"
                else:  # CO2
                    df = df_co2
                    color_scale = "Greens"
                    units = "tonnes"
                
                create_comparison_bar_chart(df, muni_col, selected_municipalities, latest_year, pollutant_type, color_scale, units)
                
                # Add historical trend chart
                st.markdown(f"### {pollutant_type} Historical Trend")
                create_historical_trend(df, muni_col, selected_municipalities, year_range, pollutant_type, units)
                
        except Exception as e:
            st.error(f"Error creating comparison charts: {e}")
    
    # Maps Tab
    with tab3:
        st.markdown('<p class="sub-header">Geographic Distribution</p>', unsafe_allow_html=True)
        
        # Select which datasets to map based on pollutant selection
        if pollutant_type == 'PM10' or pollutant_type == 'All':
            st.markdown("### PM10 Emissions Map")
            map_obj, error = create_choropleth_map(df_pm10, map_year, "PM10", italy_geojson)
            if map_obj:
                folium_static(map_obj, width=900, height=500)
            elif error:
                st.error(error)
                
        if pollutant_type == 'PM2.5' or pollutant_type == 'All':
            st.markdown("### PM2.5 Emissions Map")
            map_obj, error = create_choropleth_map(df_pm25, map_year, "PM2.5", italy_geojson)
            if map_obj:
                folium_static(map_obj, width=900, height=500)
            elif error:
                st.error(error)
                
        if pollutant_type == 'CO2' or pollutant_type == 'All':
            st.markdown("### CO2 Emissions Map")
            map_obj, error = create_choropleth_map(df_co2, map_year, "CO2", italy_geojson)
            if map_obj:
                folium_static(map_obj, width=900, height=500)
            elif error:
                st.error(error)
    
    # Statistics Tab
    with tab4:
        st.markdown('<p class="sub-header">Statistical Summary</p>', unsafe_allow_html=True)
        
        try:
            # Determine municipality column name
            muni_col = 'COMUNE' if 'COMUNE' in df_pm10.columns else 'Municipality'
            
            # Create statistics tables for selected pollutants
            if pollutant_type in ['PM10', 'All']:
                create_statistics_summary(df_pm10, muni_col, selected_municipalities, year_range, "PM10")
                
            if pollutant_type in ['PM2.5', 'All']:
                create_statistics_summary(df_pm25, muni_col, selected_municipalities, year_range, "PM2.5")
                
            if pollutant_type in ['CO2', 'All']:
                create_statistics_summary(df_co2, muni_col, selected_municipalities, year_range, "CO2")
                
            # If all pollutants selected, add correlation analysis
            if pollutant_type == 'All':
                st.markdown('<p class="sub-header">Correlation Analysis</p>', unsafe_allow_html=True)
                create_correlation_analysis(df_pm10, df_pm25, df_co2, muni_col, selected_municipalities, year_range)
                
        except Exception as e:
            st.error(f"Error in statistical analysis: {e}")

# Add instructions for using the dashboard
st.markdown("""
---
## How to Use This Dashboard

### Data Requirements
This dashboard is designed to work with CSV files that have the following structure:
- A column named `COMUNE` that contains municipality names
- Year columns with the format `emissions_YYYY` (e.g., `emissions_2011`, `emissions_2012`, etc.)
- Optional latitude (`lat`) and longitude (`lon`) columns for mapping

### Steps to Run
1. Upload your PM10, PM2.5, and CO2 data files using the sidebar uploaders
2. Use the filters to select specific municipalities, years, and pollutant types
3. Explore the various tabs for different types of analysis:
   - Time Series: View trends over time
   - Comparisons: Compare municipalities and pollutants
   - Maps: Visualize geographical distribution
   - Statistics: Get detailed statistical analysis

### Adding More Data
- To add CO2 data, create a CSV file with the same structure as your PM data
- To improve maps, add latitude and longitude columns to your data files
- For best results, ensure consistent municipality names across all data files

For detailed error messages and debugging help, expand the error messages that appear.
""")
def load_italy_geojson():
    try:
        # Check if the file exists locally
        if os.path.exists('Data/italy_municipalities.geojson'):
            with open('Data/italy_municipalities.geojson', 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            return geojson_data
        else:
            # Create a simplified version for demonstration
            st.warning("GeoJSON file not found. Using simplified geometry for demonstration.")
            return None
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return None

# Function to create choropleth map
def create_choropleth_map(df, year, pollutant_type, geojson_data=None):
    try:
        # Determine the municipality column name
        muni_col = 'COMUNE' if 'COMUNE' in df.columns else 'Municipality'
        
        # Check if latitude and longitude columns exist
        has_coordinates = 'lat' in df.columns and 'lon' in df.columns
        
        # Get the emissions column for the selected year
        emissions_col = f'emissions_{year}'
        
        if emissions_col not in df.columns:
            return None, "Selected year data not available"
        
        # Create a map centered on Italy
        m = folium.Map(location=[42.5, 12.5], zoom_start=6, 
                      tiles='CartoDB positron', control_scale=True)
        
        # Add title to map
        title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{pollutant_type} Emissions in {year}</b></h3>
             '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        if geojson_data and 'features' in geojson_data:
            # Create a choropleth map using geojson
            max_value = df[emissions_col].max()
            
            # Create a choropleth layer
            folium.Choropleth(
                geo_data=geojson_data,
                name=f'{pollutant_type} Emissions',
                data=df,
                columns=[muni_col, emissions_col],
                key_on=f"feature.properties.name",  # Adjust based on your GeoJSON structure
                fill_color="YlOrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=f"{pollutant_type} Emissions ({year})",
                highlight=True
            ).add_to(m)
        
        # Even without GeoJSON, we can add markers if coordinates are available
        if has_coordinates:
            # Create a feature group for markers
            marker_group = folium.FeatureGroup(name=f"{pollutant_type} Markers")
            
            # Add markers for each municipality with emissions data
            for idx, row in df.iterrows():
                if pd.notna(row[emissions_col]):
                    # Determine marker color based on emissions value
                    emissions_value = row[emissions_col]
                    if pollutant_type == 'CO2':
                        # Different scale for CO2
                        if emissions_value > 8000:
                            color = 'red'
                        elif emissions_value > 5000:
                            color = 'orange'
                        else:
                            color = 'green'
                    else:
                        # Scale for PM10 and PM2.5
                        if emissions_value > 35:
                            color = 'red'
                        elif emissions_value > 20:
                            color = 'orange'
                        else:
                            color = 'green'
                    
                    # Create popup content
                    popup_content = f"""
                    <b>{row[muni_col]}</b><br>
                    {pollutant_type} Emissions: {emissions_value:.2f}
                    """
                    
                    # Add marker
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']],
                        radius=max(5, min(15, emissions_value/10 if pollutant_type != 'CO2' else emissions_value/1000)),
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_content, max_width=200)
                    ).add_to(marker_group)
            
            marker_group.add_to(m)
            
            # Add heatmap layer
            heat_data = [[row['lat'], row['lon'], row[emissions_col]] 
                         for idx, row in df.iterrows() if pd.notna(row[emissions_col])]
            
            if heat_data:
                HeatMap(heat_data, radius=15, blur=10, 
                       gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 1: 'red'}).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m, None
    except Exception as e:
        return None, f"Error creating map: {e}"

# Helper function for creating bar charts in the comparison tab
def create_comparison_bar_chart(df, muni_col, selected_municipalities, year, pollutant_type, color_scale, units="μg/m³"):
    # Check for emissions_YYYY format first
    year_col = f'emissions_{year}'
    
    # If not found, try other formats
    if year_col not in df.columns:
        if str(year) in df.columns:
            year_col = str(year)
        elif f'{pollutant_type}_{year}' in df.columns:
            year_col = f'{pollutant_type}_{year}'
            
    if year_col in df.columns:
        latest_data = df[df[muni_col].isin(selected_municipalities)]
        latest_data = latest_data[[muni_col, year_col]].sort_values(by=year_col, ascending=False)
        
        if not latest_data.empty:
            # Create bar chart
            fig_bar = px.bar(
                latest_data,
                x=muni_col,
                y=year_col,
                title=f'{pollutant_type} Levels by Municipality ({year})',
                color=year_col,
                color_continuous_scale=color_scale,
                labels={year_col: f'{pollutant_type} ({units})', muni_col: 'Municipality'}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info(f"No {pollutant_type} data available for selected municipalities in {year}")
    else:
        st.info(f"Column for year {year} not found in {pollutant_type} data")
        # Debug info
        with st.expander(f"Debug - Available Columns for {pollutant_type}"):
            st.write(df.columns.tolist())

# Helper function for creating historical trend charts
def create_historical_trend(df, muni_col, selected_municipalities, year_range, pollutant_type, units="μg/m³"):
    try:
        # Filter for selected municipalities
        selected_data = df[df[muni_col].isin(selected_municipalities)]
        
        # Identify emissions columns
        emissions_cols = [col for col in selected_data.columns if col.startswith('emissions_')]
        
        if emissions_cols:
            # Calculate yearly averages
            yearly_avg = {}
            for col in emissions_cols:
                year = int(col.replace('emissions_', ''))
                if year_range[0] <= year <= year_range[1]:
                    yearly_avg[year] = selected_data[col].mean()
            
            if yearly_avg:
                trend_df = pd.DataFrame({
                    'Year': list(yearly_avg.keys()),
                    f'Average {pollutant_type}': list(yearly_avg.values())
                })
                
                # Sort by year
                trend_df = trend_df.sort_values('Year')
                
                # Create area chart for trend
                if pollutant_type == "PM10":
                    color = 'rgba(239, 85, 59, 0.7)'
                elif pollutant_type == "PM2.5":
                    color = 'rgba(255, 127, 14, 0.7)'
                else:  # CO2
                    color = 'rgba(44, 160, 44, 0.7)'
                
                fig_trend = px.area(
                    trend_df,
                    x='Year',
                    y=f'Average {pollutant_type}',
                    title=f'Average {pollutant_type} Trend',
                    labels={f'Average {pollutant_type}': f'{pollutant_type} ({units})'},
                    color_discrete_sequence=[color]
                )
                fig_trend.update_layout(height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No data available for the selected year range")
        else:
            st.info("Could not identify emissions columns for trend analysis")
    except Exception as e:
        st.error(f"Error in {pollutant_type} trend analysis: {e}")

# Helper function for creating combined trend analysis
def create_trend_chart(df_pm10, df_pm25, df_co2, muni_col, selected_municipalities, year_range):
    try:
        # Calculate yearly averages for each pollutant type
        trend_data = {
            'Year': [],
            'PM10': [],
            'PM2.5': [],
            'CO2 (scaled)': []  # CO2 will be scaled down to fit on the same graph
        }
        
        # Process the years in the range
        for year in range(year_range[0], year_range[1] + 1):
            trend_data['Year'].append(year)
            
            # PM10
            pm10_col = f'emissions_{year}'
            if pm10_col in df_pm10.columns:
                pm10_filtered = df_pm10[df_pm10[muni_col].isin(selected_municipalities)]
                trend_data['PM10'].append(pm10_filtered[pm10_col].mean())
            else:
                trend_data['PM10'].append(None)
                
            # PM2.5
            pm25_col = f'emissions_{year}'
            if pm25_col in df_pm25.columns:
                pm25_filtered = df_pm25[df_pm25[muni_col].isin(selected_municipalities)]
                trend_data['PM2.5'].append(pm25_filtered[pm25_col].mean())
            else:
                trend_data['PM2.5'].append(None)
                
            # CO2 (scaled down for comparison)
            co2_col = f'emissions_{year}'
            if co2_col in df_co2.columns:
                co2_filtered = df_co2[df_co2[muni_col].isin(selected_municipalities)]
                # Scale CO2 down by 100 or more for comparison
                scaled_co2 = co2_filtered[co2_col].mean() / 200
                trend_data['CO2 (scaled)'].append(scaled_co2)
            else:
                trend_data['CO2 (scaled)'].append(None)
        
        # Create dataframe for plotting
        trend_df = pd.DataFrame(trend_data)
        
        # Plot combined trend
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_df['Year'], y=trend_df['PM10'],
            mode='lines+markers',
            name='PM10 (μg/m³)',
            line=dict(color='rgba(239, 85, 59, 0.9)', width=2),
            marker=dict(size=8, symbol='circle')
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_df['Year'], y=trend_df['PM2.5'],
            mode='lines+markers',
            name='PM2.5 (μg/m³)',
            line=dict(color='rgba(255, 127, 14, 0.9)', width=2),
            marker=dict(size=8, symbol='square')
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_df['Year'], y=trend_df['CO2 (scaled)'],
            mode='lines+markers',
            name='CO2 (scaled 1:200)',
            line=dict(color='rgba(44, 160, 44, 0.9)', width=2),
            marker=dict(size=8, symbol='diamond')
        ))
        
        fig.update_layout(
            title='Comparative Trend Analysis (2011-2022)',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        *Note: CO2 values are scaled down by a factor of 200 to allow for visual comparison with PM10 and PM2.5 on the same chart.
        The actual CO2 values are much higher (in tonnes).*
        """)
    except Exception as e:
        st.error(f"Error creating trend chart: {e}")

# Helper function for creating statistics tables and visualizations
def create_statistics_summary(df, muni_col, selected_municipalities, year_range, pollutant_type):
    try:
        st.markdown(f"### {pollutant_type} Statistics")
        
        # Prepare data for statistics
        stats_data = {}
        for muni in selected_municipalities:
            muni_data = df[df[muni_col] == muni]
            
            # Get emissions columns for the selected years
            yearly_values = []
            for year in range(year_range[0], year_range[1] + 1):
                col = f'emissions_{year}'
                if col in muni_data.columns:
                    if len(muni_data[col].values) > 0 and not pd.isna(muni_data[col].values[0]):
                        yearly_values.append(muni_data[col].values[0])
            
            if yearly_values:
                stats_data[muni] = {
                    'Mean': np.mean(yearly_values),
                    'Min': np.min(yearly_values),
                    'Max': np.max(yearly_values),
                    'Std Dev': np.std(yearly_values),
                    'Median': np.median(yearly_values)
                }
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data).T
            stats_df = stats_df.sort_values(by='Mean', ascending=False)
            
            # Display statistics table
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
            
            # Add a box plot for distribution
            box_data = []
            for muni in selected_municipalities:
                muni_data = df[df[muni_col] == muni]
                for year in range(year_range[0], year_range[1] + 1):
                    col = f'emissions_{year}'
                    if col in muni_data.columns and len(muni_data[col].values) > 0 and not pd.isna(muni_data[col].values[0]):
                        box_data.append({
                            'Municipality': muni,
                            'Value': muni_data[col].values[0],
                            'Year': year
                        })
            
            if box_data:
                box_df = pd.DataFrame(box_data)
                
                # Create box plot for distribution
                units = "tonnes" if pollutant_type == "CO2" else "μg/m³"
                fig_box = px.box(
                    box_df,
                    x='Municipality',
                    y='Value',
                    title=f'{pollutant_type} Distribution by Municipality',
                    labels={'Value': f'{pollutant_type} ({units})'}
                )
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Add a histogram for overall distribution
                fig_hist = px.histogram(
                    box_df, 
                    x='Value',
                    nbins=20,
                    title=f'Overall {pollutant_type} Distribution',
                    labels={'Value': f'{pollutant_type} ({units})'}
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No data available for statistical analysis with the selected parameters")
    except Exception as e:
        st.error(f"Error in {pollutant_type} statistical analysis: {e}")

# Helper function for creating PM-CO2 correlations
def create_pollutant_co2_correlation(df_pm, df_co2, muni_col, selected_municipalities, year_range, pm_type):
    try:
        corr_data = []
        for muni in selected_municipalities:
            pm_row = df_pm[df_pm[muni_col] == muni]
            co2_row = df_co2[df_co2[muni_col] == muni]
            
            for year in range(year_range[0], year_range[1] + 1):
                pm_col = f'emissions_{year}'
                co2_col = f'emissions_{year}'
                
                if pm_col in pm_row.columns and co2_col in co2_row.columns:
                    if len(pm_row[pm_col].values) > 0 and len(co2_row[co2_col].values) > 0:
                        pm_val = pm_row[pm_col].values[0]
                        co2_val = co2_row[co2_col].values[0]
                        
                        if not pd.isna(pm_val) and not pd.isna(co2_val):
                            corr_data.append({
                                'Municipality': muni,
                                'Year': year,
                                pm_type: pm_val,
                                'CO2': co2_val
                            })
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            
            # Create scatter plot
            fig_corr = px.scatter(
                corr_df,
                x='CO2',
                y=pm_type,
                color='Municipality',
                hover_data=['Year'],
                title=f'{pm_type} vs CO2 Correlation',
                trendline='ols',
                labels={
                    'CO2': 'CO2 (tonnes)',
                    pm_type: f'{pm_type} (μg/m³)'
                }
            )
            fig_corr.update_layout(height=350)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Calculate correlation coefficient
            corr_value = corr_df[pm_type].corr(corr_df['CO2'])
            st.info(f"{pm_type}-CO2 Correlation Coefficient: {corr_value:.3f}")
        else:
            st.info(f"Insufficient data for {pm_type}-CO2 correlation analysis.")
    except Exception as e:
        st.error(f"Error in {pm_type}-CO2 correlation analysis: {e}")

# Helper function for creating correlation analysis
def create_correlation_analysis(df_pm10, df_pm25, df_co2, muni_col, selected_municipalities, year_range):
    try:
        # Prepare data for correlations
        col1, col2 = st.columns(2)
        
        with col1:
            # PM10 vs PM2.5 Correlation
            st.markdown("### PM10 vs PM2.5 Correlation")
            
            pm_corr_data = []
            for muni in selected_municipalities:
                pm10_row = df_pm10[df_pm10[muni_col] == muni]
                pm25_row = df_pm25[df_pm25[muni_col] == muni]
                
                for year in range(year_range[0], year_range[1] + 1):
                    pm10_col = f'emissions_{year}'
                    pm25_col = f'emissions_{year}'
                    
                    if pm10_col in pm10_row.columns and pm25_col in pm25_row.columns:
                        if len(pm10_row[pm10_col].values) > 0 and len(pm25_row[pm25_col].values) > 0:
                            pm10_val = pm10_row[pm10_col].values[0]
                            pm25_val = pm25_row[pm25_col].values[0]
                            
                            if not pd.isna(pm10_val) and not pd.isna(pm25_val):
                                pm_corr_data.append({
                                    'Municipality': muni,
                                    'Year': year,
                                    'PM10': pm10_val,
                                    'PM2.5': pm25_val
                                })
            
            if pm_corr_data:
                pm_corr_df = pd.DataFrame(pm_corr_data)
                
                # Create scatter plot
                fig_pm_corr = px.scatter(
                    pm_corr_df,
                    x='PM10',
                    y='PM2.5',
                    color='Municipality',
                    hover_data=['Year'],
                    title='PM10 vs PM2.5 Correlation',
                    trendline='ols',
                    labels={
                        'PM10': 'PM10 (μg/m³)',
                        'PM2.5': 'PM2.5 (μg/m³)'
                    }
                )
                fig_pm_corr.update_layout(height=400)
                st.plotly_chart(fig_pm_corr, use_container_width=True)
                
                # Calculate correlation coefficient
                corr_value = pm_corr_df['PM10'].corr(pm_corr_df['PM2.5'])
                st.info(f"PM10-PM2.5 Correlation Coefficient: {corr_value:.3f}")
            else:
                st.info("Insufficient data for PM10-PM2.5 correlation analysis.")
        
        with col2:
            # PM10/PM2.5 vs CO2 Correlation
            st.markdown("### Particulate Matter vs CO2 Correlation")
            
            # Create tabs for PM10-CO2 and PM2.5-CO2 correlations
            pm_co2_tab1, pm_co2_tab2 = st.tabs(["PM10 vs CO2", "PM2.5 vs CO2"])
            
            with pm_co2_tab1:
                create_pollutant_co2_correlation(df_pm10, df_co2, muni_col, selected_municipalities, year_range, "PM10")
                
            with pm_co2_tab2:
                create_pollutant_co2_correlation(df_pm25, df_co2, muni_col, selected_municipalities, year_range, "PM2.5")
    
    except Exception as e:
        st.error(f"Error in correlation analysis: {e}")

# Function to load data
#@st.cache_data
