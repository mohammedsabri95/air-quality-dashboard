import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
st.markdown('<p class="main-header">Italian Municipalities Air Quality Dashboard (2011-2022)</p>', unsafe_allow_html=True)

# Display app information
with st.expander("About This Dashboard"):
    st.markdown("""
    This dashboard visualizes PM10 and PM2.5 air quality data for Italian municipalities from 2011 to 2022.
    - Upload your CSV files using the sidebar uploader
    - Filter by municipality, year range, and pollutant type
    - Explore different visualizations including time series, comparisons, and statistical summaries
    
    If you encounter any issues with your data format, check the debug information in the sidebar.
    """)

def create_sample_data_with_emissions_format(pollutant_type):
    """Create sample data for demonstration using the emissions_YYYY format"""
    municipalities = ["Milan", "Rome", "Naples", "Turin", "Florence", 
                      "Bologna", "Venice", "Genoa", "Palermo", "Bari"]
    
    # Create emissions_YYYY columns (2011 to 2022)
    data = {}
    for year in range(2011, 2023):
        # Different base values for PM10 vs PM2.5
        base_value = 40 if pollutant_type == "PM10" else 25
        # Generate random data with a decreasing trend
        data[f'emissions_{year}'] = np.random.normal(
            base_value - (year - 2011) * (1.5 if pollutant_type == "PM10" else 1.0), 
            10 if pollutant_type == "PM10" else 8, 
            size=len(municipalities)
        )
    
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
            year_columns = [col for col in df_copy.columns if col != municipality_col and pd.api.types.is_numeric_dtype(df_copy[col])]
    
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
        if 'emissions_' in df_melted['Year'].iloc[0]:
            df_melted['Year'] = df_melted['Year'].str.replace('emissions_', '').astype(int)
        elif '_' in df_melted['Year'].iloc[0]:
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

# Function to load data
@st.cache_data
def load_data():
    try:
        # Add file uploader for PM10 and PM2.5 data
        st.sidebar.markdown("### Upload Data Files")
        pm10_file = st.sidebar.file_uploader("Upload PM10 CSV file", type=['csv'])
        pm25_file = st.sidebar.file_uploader("Upload PM2.5 CSV file", type=['csv'])
        
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
        
        # Display data structure information
        with st.sidebar.expander("View Data Structure"):
            st.markdown("#### PM10 DataFrame Columns")
            st.write(df_pm10.columns.tolist())
            st.markdown("#### PM2.5 DataFrame Columns")
            st.write(df_pm25.columns.tolist())
        
        return df_pm10, df_pm25
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Fallback to sample data with the expected column format
        return create_sample_data_with_emissions_format("PM10"), create_sample_data_with_emissions_format("PM2.5")

# Load the data
# Load the data
df_pm25 = pd.read_csv('total_PM2.5_2011_2022_whole_italy_emissions.csv')
df_pm10 = pd.read_csv('total_PM10_2011_2022_whole_italy_emissions.csv')

# Sidebar for filtering
st.sidebar.markdown('## Filters')

# Detect min and max years from data based on emissions_YYYY format
pm10_year_cols = [col for col in df_pm10.columns if col.startswith('emissions_')]
pm25_year_cols = [col for col in df_pm25.columns if col.startswith('emissions_')]

# Extract years from emissions_YYYY format
pm10_years = []
if pm10_year_cols:
    pm10_years = [int(col.replace('emissions_', '')) for col in pm10_year_cols]

pm25_years = []
if pm25_year_cols:
    pm25_years = [int(col.replace('emissions_', '')) for col in pm25_year_cols]

# If no emissions columns found, try other formats
if not pm10_years:
    # Try direct year columns
    direct_year_cols = [col for col in df_pm10.columns if col.isdigit() and 2000 <= int(col) <= 2030]
    if direct_year_cols:
        pm10_years = [int(col) for col in direct_year_cols]
    # Try PM10_YYYY format
    elif any('PM10_' in col for col in df_pm10.columns):
        pm10_year_cols = [col for col in df_pm10.columns if 'PM10_' in col]
        pm10_years = [int(col.split('_')[1]) for col in pm10_year_cols]

if not pm25_years:
    # Try direct year columns
    direct_year_cols = [col for col in df_pm25.columns if col.isdigit() and 2000 <= int(col) <= 2030]
    if direct_year_cols:
        pm25_years = [int(col) for col in direct_year_cols]
    # Try PM2.5_YYYY format
    elif any('PM2.5_' in col for col in df_pm25.columns):
        pm25_year_cols = [col for col in df_pm25.columns if 'PM2.5_' in col]
        pm25_years = [int(col.split('_')[1]) for col in pm25_year_cols]

# Combine all detected years
all_years = pm10_years + pm25_years

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

# Pollutant type selection
pollutant_type = st.sidebar.radio(
    'Select Pollutant Type',
    options=['PM10', 'PM2.5', 'Both']
)

# Main dashboard content
if not selected_municipalities:
    st.warning("Please select at least one municipality from the sidebar to display visualizations.")
else:
    # Create time series section
    col1, col2 = st.columns(2)

    # Time Series Analysis
    with col1:
        st.markdown('<p class="sub-header">Time Series Analysis</p>', unsafe_allow_html=True)
        
        try:
            # Filter data based on selections
            if pollutant_type in ['PM10', 'Both']:
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
            
            if pollutant_type in ['PM2.5', 'Both']:
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
        except Exception as e:
            st.error(f"Error creating time series plots: {e}")

    # Comparison between municipalities
    with col2:
        st.markdown('<p class="sub-header">Municipality Comparison</p>', unsafe_allow_html=True)
        
        try:
            # Select most recent year in the range
            latest_year = min(max_year, year_range[1])
            
            # Determine municipality column name
            muni_col = 'COMUNE' if 'COMUNE' in df_pm10.columns else 'Municipality'
            
            # Filter data for the latest year and selected municipalities
            if pollutant_type in ['PM10', 'Both']:
                # Check for emissions_YYYY format first
                year_col = f'emissions_{latest_year}'
                
                # If not found, try other formats
                if year_col not in df_pm10.columns:
                    if str(latest_year) in df_pm10.columns:
                        year_col = str(latest_year)
                    elif f'PM10_{latest_year}' in df_pm10.columns:
                        year_col = f'PM10_{latest_year}'
                        
                if year_col in df_pm10.columns:
                    latest_pm10 = df_pm10[df_pm10[muni_col].isin(selected_municipalities)]
                    latest_pm10 = latest_pm10[[muni_col, year_col]].sort_values(by=year_col, ascending=False)
                    
                    # Create bar chart for PM10
                    fig_bar_pm10 = px.bar(
                        latest_pm10,
                        x=muni_col,
                        y=year_col,
                        title=f'PM10 Levels by Municipality ({latest_year})',
                        color=year_col,
                        color_continuous_scale='Reds',
                        labels={year_col: 'PM10 (μg/m³)', muni_col: 'Municipality'}
                    )
                    fig_bar_pm10.update_layout(height=400)
                    st.plotly_chart(fig_bar_pm10, use_container_width=True)
                else:
                    st.info(f"Column for year {latest_year} not found in PM10 data")
                    # Debug info
                    with st.expander("Debug - Available Columns"):
                        st.write(df_pm10.columns.tolist())
            
            if pollutant_type in ['PM2.5', 'Both']:
                # Check for emissions_YYYY format first
                year_col = f'emissions_{latest_year}'
                
                # If not found, try other formats
                if year_col not in df_pm25.columns:
                    if str(latest_year) in df_pm25.columns:
                        year_col = str(latest_year)
                    elif f'PM2.5_{latest_year}' in df_pm25.columns:
                        year_col = f'PM2.5_{latest_year}'
                        
                if year_col in df_pm25.columns:
                    latest_pm25 = df_pm25[df_pm25[muni_col].isin(selected_municipalities)]
                    latest_pm25 = latest_pm25[[muni_col, year_col]].sort_values(by=year_col, ascending=False)
                    
                    # Create bar chart for PM2.5
                    fig_bar_pm25 = px.bar(
                        latest_pm25,
                        x=muni_col,
                        y=year_col,
                        title=f'PM2.5 Levels by Municipality ({latest_year})',
                        color=year_col,
                        color_continuous_scale='Oranges',
                        labels={year_col: 'PM2.5 (μg/m³)', muni_col: 'Municipality'}
                    )
                    fig_bar_pm25.update_layout(height=400)
                    st.plotly_chart(fig_bar_pm25, use_container_width=True)
                else:
                    st.info(f"Column for year {latest_year} not found in PM2.5 data")
                    # Debug info
                    with st.expander("Debug - Available Columns"):
                        st.write(df_pm25.columns.tolist())
        except Exception as e:
            st.error(f"Error creating municipality comparison: {e}")

    # Create 2 more sections
    col3, col4 = st.columns(2)

    # Historical Trends Analysis
    with col3:
        st.markdown('<p class="sub-header">Historical Trends Analysis</p>', unsafe_allow_html=True)
        
        # Calculate average PM levels per year across selected municipalities
        if pollutant_type in ['PM10', 'Both']:
            try:
                # Determine municipality column name
                muni_col = 'COMUNE' if 'COMUNE' in df_pm10.columns else 'Municipality'
                
                # Filter for selected municipalities
                pm10_selected = df_pm10[df_pm10[muni_col].isin(selected_municipalities)]
                
                # Identify emissions columns
                emissions_cols = [col for col in pm10_selected.columns if col.startswith('emissions_')]
                
                if emissions_cols:
                    # Calculate yearly averages
                    pm10_yearly_avg = {}
                    for col in emissions_cols:
                        year = int(col.replace('emissions_', ''))
                        if year_range[0] <= year <= year_range[1]:
                            pm10_yearly_avg[year] = pm10_selected[col].mean()
                    
                    if pm10_yearly_avg:
                        pm10_trend_df = pd.DataFrame({
                            'Year': list(pm10_yearly_avg.keys()),
                            'Average PM10': list(pm10_yearly_avg.values())
                        })
                        
                        # Sort by year
                        pm10_trend_df = pm10_trend_df.sort_values('Year')
                        
                        # Create area chart for PM10 trend
                        fig_trend_pm10 = px.area(
                            pm10_trend_df,
                            x='Year',
                            y='Average PM10',
                            title='Average PM10 Trend',
                            labels={'Average PM10': 'PM10 (μg/m³)'},
                            color_discrete_sequence=['rgba(239, 85, 59, 0.7)']
                        )
                        fig_trend_pm10.update_layout(height=400)
                        st.plotly_chart(fig_trend_pm10, use_container_width=True)
                    else:
                        st.info("No data available for the selected year range")
                else:
                    st.info("Could not identify emissions columns for trend analysis")
            except Exception as e:
                st.error(f"Error in PM10 trend analysis: {e}")
        
        if pollutant_type in ['PM2.5', 'Both']:
            try:
                # Determine municipality column name
                muni_col = 'COMUNE' if 'COMUNE' in df_pm25.columns else 'Municipality'
                
                # Filter for selected municipalities
                pm25_selected = df_pm25[df_pm25[muni_col].isin(selected_municipalities)]
                
                # Identify emissions columns
                emissions_cols = [col for col in pm25_selected.columns if col.startswith('emissions_')]
                
                if emissions_cols:
                    # Calculate yearly averages
                    pm25_yearly_avg = {}
                    for col in emissions_cols:
                        year = int(col.replace('emissions_', ''))
                        if year_range[0] <= year <= year_range[1]:
                            pm25_yearly_avg[year] = pm25_selected[col].mean()
                    
                    if pm25_yearly_avg:
                        pm25_trend_df = pd.DataFrame({
                            'Year': list(pm25_yearly_avg.keys()),
                            'Average PM2.5': list(pm25_yearly_avg.values())
                        })
                        
                        # Sort by year
                        pm25_trend_df = pm25_trend_df.sort_values('Year')
                        
                        # Create area chart for PM2.5 trend
                        fig_trend_pm25 = px.area(
                            pm25_trend_df,
                            x='Year',
                            y='Average PM2.5',
                            title='Average PM2.5 Trend',
                            labels={'Average PM2.5': 'PM2.5 (μg/m³)'},
                            color_discrete_sequence=['rgba(255, 127, 14, 0.7)']
                        )
                        fig_trend_pm25.update_layout(height=400)
                        st.plotly_chart(fig_trend_pm25, use_container_width=True)
                    else:
                        st.info("No data available for the selected year range")
                else:
                    st.info("Could not identify emissions columns for trend analysis")
            except Exception as e:
                st.error(f"Error in PM2.5 trend analysis: {e}")

    # Statistical Summary
    with col4:
        st.markdown('<p class="sub-header">Statistical Summary</p>', unsafe_allow_html=True)
        
        # Create statistics tables for the selected year range
        if pollutant_type in ['PM10', 'Both']:
            try:
                # Determine municipality column name
                muni_col = 'COMUNE' if 'COMUNE' in df_pm10.columns else 'Municipality'
                
                # Prepare data for statistics
                pm10_stats_data = {}
                for muni in selected_municipalities:
                    muni_data = df_pm10[df_pm10[muni_col] == muni]
                    
                    # Get emissions columns for the selected years
                    yearly_values = []
                    for year in range(year_range[0], year_range[1] + 1):
                        col = f'emissions_{year}'
                        if col in muni_data.columns:
                            if not pd.isna(muni_data[col].values[0]):
                                yearly_values.append(muni_data[col].values[0])
                    
                    if yearly_values:
                        pm10_stats_data[muni] = {
                            'Mean': np.mean(yearly_values),
                            'Min': np.min(yearly_values),
                            'Max': np.max(yearly_values),
                            'Std Dev': np.std(yearly_values)
                        }
                
                if pm10_stats_data:
                    pm10_stats_df = pd.DataFrame(pm10_stats_data).T
                    pm10_stats_df = pm10_stats_df.sort_values(by='Mean', ascending=False)
                    
                    st.markdown("### PM10 Statistics")
                    st.dataframe(pm10_stats_df.style.format("{:.2f}"), use_container_width=True)
                    
                    # Add a box plot
                    pm10_box_data = []
                    for muni in selected_municipalities:
                        muni_data = df_pm10[df_pm10[muni_col] == muni]
                        for year in range(year_range[0], year_range[1] + 1):
                            col = f'emissions_{year}'
                            if col in muni_data.columns and not pd.isna(muni_data[col].values[0]):
                                pm10_box_data.append({
                                    'Municipality': muni,
                                    'Value': muni_data[col].values[0],
                                    'Year': year
                                })
                    
                    if pm10_box_data:
                        pm10_box_df = pd.DataFrame(pm10_box_data)
                        fig_box_pm10 = px.box(
                            pm10_box_df,
                            x='Municipality',
                            y='Value',
                            title='PM10 Distribution by Municipality',
                            labels={'Value': 'PM10 (μg/m³)'}
                        )
                        fig_box_pm10.update_layout(height=400)
                        st.plotly_chart(fig_box_pm10, use_container_width=True)
                else:
                    st.info("No data available for statistical analysis with the selected parameters")
            except Exception as e:
                st.error(f"Error in PM10 statistical analysis: {e}")
        
        if pollutant_type in ['PM2.5', 'Both']:
            try:
                # Determine municipality column name
                muni_col = 'COMUNE' if 'COMUNE' in df_pm25.columns else 'Municipality'
                
                # Prepare data for statistics
                pm25_stats_data = {}
                for muni in selected_municipalities:
                    muni_data = df_pm25[df_pm25[muni_col] == muni]
                    
                    # Get emissions columns for the selected years
                    yearly_values = []
                    for year in range(year_range[0], year_range[1] + 1):
                        col = f'emissions_{year}'
                        if col in muni_data.columns:
                            if not pd.isna(muni_data[col].values[0]):
                                yearly_values.append(muni_data[col].values[0])
                    
                    if yearly_values:
                        pm25_stats_data[muni] = {
                            'Mean': np.mean(yearly_values),
                            'Min': np.min(yearly_values),
                            'Max': np.max(yearly_values),
                            'Std Dev': np.std(yearly_values)
                        }
                
                if pm25_stats_data:
                    pm25_stats_df = pd.DataFrame(pm25_stats_data).T
                    pm25_stats_df = pm25_stats_df.sort_values(by='Mean', ascending=False)
                    
                    st.markdown("### PM2.5 Statistics")
                    st.dataframe(pm25_stats_df.style.format("{:.2f}"), use_container_width=True)
                else:
                    st.info("No data available for statistical analysis with the selected parameters")
            except Exception as e:
                st.error(f"Error in PM2.5 statistical analysis: {e}")

    # Add correlation analysis section at the bottom if both pollutant types are selected
    if pollutant_type == 'Both':
        st.markdown('<p class="sub-header">PM10 vs PM2.5 Correlation Analysis</p>', unsafe_allow_html=True)
        
        try:
            # Determine municipality column name
            muni_col = 'COMUNE' if 'COMUNE' in df_pm10.columns else 'Municipality'
            
            # Prepare data for correlation
            corr_data = []
            for muni in selected_municipalities:
                pm10_row = df_pm10[df_pm10[muni_col] == muni]
                pm25_row = df_pm25[df_pm25[muni_col] == muni]
                
                for year in range(year_range[0], year_range[1] + 1):
                    pm10_col = f'emissions_{year}'
                    pm25_col = f'emissions_{year}'
                    
                    if pm10_col in pm10_row.columns and pm25_col in pm25_row.columns:
                        pm10_val = pm10_row[pm10_col].values[0]
                        pm25_val = pm25_row[pm25_col].values[0]
                        
                        if not pd.isna(pm10_val) and not pd.isna(pm25_val):
                            corr_data.append({
                                'Municipality': muni,
                                'Year': year,
                                'PM10': pm10_val,
                                'PM2.5': pm25_val
                            })
            
            if corr_data:
                corr_df = pd.DataFrame(corr_data)
                
                # Create scatter plot
                fig_corr = px.scatter(
                    corr_df,
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
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Calculate overall correlation
                corr_value = corr_df['PM10'].corr(corr_df['PM2.5'])
                st.info(f"Overall Correlation Coefficient: {corr_value:.3f}")
            else:
                st.info("Insufficient data for correlation analysis. Please ensure both PM10 and PM2.5 datasets have matching years and municipalities.")
        except Exception as e:
            st.error(f"Error in correlation analysis: {e}")


