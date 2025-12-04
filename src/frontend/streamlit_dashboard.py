"""
Fishery Disaster Appropriations Dashboard

Interactive dashboard for exploring:
- Spatial fishery data with heatwave overlays
- Historical disasters
- Custom climate scenarios
- Monte Carlo predictions
- Distribution visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
import traceback
import warnings
warnings.filterwarnings('ignore')

# Try to import xarray for NetCDF files
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

# Set page config
st.set_page_config(
    page_title="Fishery Disaster Risk Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Paths
DATA_DIR = Path('data')
CSV_DIR = DATA_DIR / 'csv'
SHAPEFILES_DIR = DATA_DIR / 'shapefiles'
HEATWAVE_DIR = DATA_DIR / 'heatwaves'
OUTPUT_DIR = DATA_DIR / 'output'

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def load_model_data():
    """Load the main model dataset."""
    df = pd.read_csv(CSV_DIR / 'model_data.csv')
    df = df.dropna(subset=['log_appropriation', 'log_total_value', 'peak_intensity',
                           'duration_days', 'percent_in_heatwave'])
    df = df[(df['peak_intensity'] > 0) & (df['log_appropriation'] > 0)]
    
    # Convert log_appropriation back to total_appropriation if not present
    if 'total_appropriation' not in df.columns:
        df['total_appropriation'] = np.exp(df['log_appropriation'])
    
    return df

@st.cache_data
def load_fitted_distributions():
    """Load fitted distributions."""
    with open(OUTPUT_DIR / 'fitted_distributions.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_posterior_samples():
    """Load posterior samples from Bayesian model."""
    return pd.read_csv(OUTPUT_DIR / 'posterior_samples.csv')

@st.cache_data
def load_scaling_params():
    """Load scaling parameters."""
    return pd.read_csv(OUTPUT_DIR / 'scaling_parameters.csv').iloc[0].to_dict()

@st.cache_data
def load_region_data():
    """Load available regions and species with their data paths."""
    shapefile_dict = {
        'Alaska': {
            'mask': 'src/climate/sst_data/alaska/spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv',
            'csv_output_path': 'src/climate/sst_data/alaska/alaska_mhw_events_filtered.csv'
        },
        'Washington': {
            'mask': 'src/climate/output_masks/washington_eez_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/west_coast/west_coast_mhw_events.csv',
            'csv_output_path': 'data/csv/west_coast_mhw_events_washington_eez_filtered.csv'
        },
        'Oregon': {
            'mask': 'src/climate/output_masks/oregon_eez_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/west_coast/west_coast_mhw_events.csv',
            'csv_output_path': 'data/csv/west_coast_mhw_events_oregon_eez_filtered.csv'
        },
        'California': {
            'mask': 'src/climate/output_masks/california_eez_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/west_coast/west_coast_mhw_events.csv',
            'csv_output_path': 'data/csv/west_coast_mhw_events_california_eez_filtered.csv'
        },
        'BC': {
            'mask': 'src/climate/sst_data/bc/spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/bc/bc_mhw_events.csv',
            'csv_output_path': 'data/csv/bc_mhw_events_bc_eez_filtered.csv'
        },
        'Washington_BC': {
            'mask': 'src/climate/output_masks/washington_bc_eez_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/washington_bc/west_coast_bc_mhw_events.csv',
            'csv_output_path': 'data/csv/west_coast_bc_mhw_events_washington_bc_eez_filtered.csv'
        },
        'West_Coast': {
            'mask': 'src/climate/sst_data/west_coast/spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/west_coast/west_coast_mhw_events.csv',
            'csv_output_path': 'data/csv/west_coast_mhw_events_west_coast_eez_filtered.csv'
        },
        'Tanner_Crab': {
            'mask': 'src/climate/output_masks/tanner_crab_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv',
            'csv_output_path': 'data/csv/alaska_mhw_events_tanner_crab_filtered.csv'
        },
        'Cod': {
            'mask': 'src/climate/output_masks/pcod_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv',
            'csv_output_path': 'data/csv/alaska_mhw_events_pcod_filtered.csv'
        },
        'Snow_Crab': {
            'mask': 'src/climate/output_masks/snow_crab_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv',
            'csv_output_path': 'data/csv/alaska_mhw_events_snow_crab_filtered.csv'
        },
        'RKK_SC': {
            'mask': 'src/climate/output_masks/snowcrab_rockcrab_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv',
            'csv_output_path': 'data/csv/alaska_mhw_events_rkk_sc_filtered.csv'
        },
        'Norton_Sound_RKK': {
            'mask': 'src/climate/output_masks/norton_sound_red_king_crab_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv',
            'csv_output_path': 'data/csv/alaska_mhw_events_norton_sound_rkk_filtered.csv'
        },
        'Bering_Sea_Red_King_Crab': {
            'mask': 'src/climate/output_masks/bering_sea_redking_crab_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv',
            'csv_output_path': 'data/csv/alaska_mhw_events_bering_sea_redking_crab_filtered.csv'
        },
        'Peconic_Estuary': {
            'mask': 'src/climate/output_masks/peconic_estuary_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/peconic_bay/peconic_bay_mhw_events.csv',
            'csv_output_path': 'data/csv/peconic_bay_mhw_events_peconic_estuary_filtered.csv'
        }
    }
    
    # Check which regions/species have data available
    available_regions = {}
    for name, paths in shapefile_dict.items():
        has_data = False
        data_info = {'name': name, 'has_mask': False, 'has_sst': False, 'has_output': False}
        
        if Path(paths['mask']).exists():
            data_info['has_mask'] = True
            has_data = True
        
        if Path(paths['sst_data_path']).exists():
            data_info['has_sst'] = True
            has_data = True
        
        if Path(paths['csv_output_path']).exists():
            data_info['has_output'] = True
            has_data = True
        
        if has_data:
            data_info['paths'] = paths
            available_regions[name] = data_info
    
    return available_regions

@st.cache_data
def load_shapefiles():
    """Load available shapefiles."""
    shapefile_dict = {}
    if SHAPEFILES_DIR.exists():
        for shp_file in SHAPEFILES_DIR.glob('*.shp'):
            try:
                gdf = gpd.read_file(shp_file)
                shapefile_dict[shp_file.stem] = gdf
            except:
                pass
    return shapefile_dict

@st.cache_data
def load_spatial_mask(mask_path):
    """Load spatial mask NetCDF file and extract coordinates."""
    if not HAS_XARRAY:
        return None
    
    try:
        mask_ds = xr.open_dataset(mask_path)
        
        # Try to find mask variable
        mask_var = None
        possible_mask_names = ['mask', 'spatial_mask', '__xarray_dataarray_variable__', 
                              'MASK', 'Mask', 'data']
        
        for var in possible_mask_names:
            if var in mask_ds.variables:
                mask_var = var
                break
        
        if mask_var is None and len(mask_ds.data_vars) > 0:
            # Just get the first data variable
            mask_var = list(mask_ds.data_vars)[0]
        
        if mask_var:
            mask_data = mask_ds[mask_var]
            
            # Get coordinates - try multiple possible names
            lat_coord = None
            lon_coord = None
            
            # Try latitude variations (prioritize 'latitude' and 'longitude')
            for coord in ['latitude', 'lat', 'y', 'LAT', 'LATITUDE', 'Latitude']:
                if coord in mask_data.coords or coord in mask_data.dims:
                    lat_coord = coord
                    break
            
            # Try longitude variations (prioritize 'longitude')
            for coord in ['longitude', 'lon', 'x', 'LON', 'LONGITUDE', 'Longitude']:
                if coord in mask_data.coords or coord in mask_data.dims:
                    lon_coord = coord
                    break
            
            if lat_coord and lon_coord:
                lats = mask_data.coords[lat_coord].values if lat_coord in mask_data.coords else mask_data[lat_coord].values
                lons = mask_data.coords[lon_coord].values if lon_coord in mask_data.coords else mask_data[lon_coord].values
                mask_values = mask_data.values
                
                # Count non-NaN cells (your masks use NaN for outside region)
                non_nan_count = np.sum(~np.isnan(mask_values))
                
                return {
                    'lats': lats,
                    'lons': lons,
                    'mask': mask_values,
                    'dataset': mask_ds,
                    'mask_var': mask_var,
                    'lat_coord': lat_coord,
                    'lon_coord': lon_coord,
                    'cells_inside': non_nan_count
                }
            else:
                # Return error info for debugging
                return {
                    'error': 'Could not find lat/lon coordinates',
                    'available_coords': list(mask_data.coords.keys()),
                    'available_dims': list(mask_data.dims),
                    'mask_var': mask_var
                }
        else:
            return {
                'error': 'Could not find mask variable',
                'available_vars': list(mask_ds.variables.keys())
            }
        
    except Exception as e:
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@st.cache_data
def load_heatwave_csv(csv_path):
    """Load heatwave CSV data."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except:
        return None

def sample_from_fitted_distribution(fit_info, n_samples=1):
    """Sample from a fitted distribution."""
    dist_name = fit_info['distribution']
    params = fit_info['params']
    dist = getattr(stats, dist_name)
    samples = dist.rvs(*params, size=n_samples)
    
    if 'scale' in fit_info:
        samples = samples * fit_info['scale']
    if 'truncate_0_100' in fit_info and fit_info['truncate_0_100']:
        samples = np.clip(samples, 0, 100)
    
    return samples

def run_monte_carlo_simulation(
    fitted_distributions,
    posterior_samples,
    scaling_params,
    frequency_multiplier=1.0,
    intensity_multiplier=1.0,
    duration_multiplier=1.0,
    coverage_multiplier=1.0,
    n_simulations=1000
):
    """Run Monte Carlo simulation with custom climate parameters."""
    
    # Modify distributions based on climate multipliers
    modified_distributions = {}
    
    # Log total value (unchanged)
    modified_distributions['log_total_value'] = fitted_distributions['log_total_value'].copy()
    
    # Peak intensity
    peak_dist = fitted_distributions['peak_intensity'].copy()
    if peak_dist['distribution'] == 'weibull_min':
        params = list(peak_dist['params'])
        params[1] = params[1] + (intensity_multiplier - 1.0) * peak_dist['mean']
        peak_dist['params'] = params
    modified_distributions['peak_intensity'] = peak_dist
    
    # Duration
    duration_dist = fitted_distributions['duration_days'].copy()
    if duration_dist['distribution'] in ['gamma', 'lognorm']:
        params = list(duration_dist['params'])
        params[2] = params[2] * duration_multiplier
        duration_dist['params'] = params
    modified_distributions['duration_days'] = duration_dist
    
    # Coverage
    coverage_dist = fitted_distributions['percent_in_heatwave'].copy()
    if coverage_dist['distribution'] == 'beta':
        params = list(coverage_dist['params'])
        params[0] = params[0] * (1 + (coverage_multiplier - 1.0))
        coverage_dist['params'] = params
    modified_distributions['percent_in_heatwave'] = coverage_dist
    
    # Disaster frequency
    disaster_freq = fitted_distributions.get('disaster_frequency', None)
    
    results = []
    
    for sim in range(n_simulations):
        # Sample disaster count
        if disaster_freq and disaster_freq['distribution'] == 'zt_poisson':
            lam = disaster_freq['params']['lambda'] * frequency_multiplier
            n_disasters = 0
            while n_disasters == 0:
                n_disasters = np.random.poisson(lam)
        else:
            n_disasters = max(1, int(np.round(np.random.choice([1,2,3,4,5,6]) * frequency_multiplier)))
        
        # Sample posterior
        sample_idx = np.random.randint(0, len(posterior_samples))
        posterior_draw = posterior_samples.iloc[sample_idx]
        
        beta_0 = posterior_draw['alpha']
        beta_value = posterior_draw['beta_value']
        beta_peak = posterior_draw['beta_peak']
        beta_duration = posterior_draw['beta_duration']
        beta_percent = posterior_draw['beta_percent']
        sigma = posterior_draw['sigma']
        
        # Sample disaster characteristics
        log_values = sample_from_fitted_distribution(
            modified_distributions['log_total_value'], n_disasters
        )
        peak_intensities = sample_from_fitted_distribution(
            modified_distributions['peak_intensity'], n_disasters
        )
        durations = sample_from_fitted_distribution(
            modified_distributions['duration_days'], n_disasters
        )
        durations = np.clip(durations, 0, 2_600_000)
        
        percents = sample_from_fitted_distribution(
            modified_distributions['percent_in_heatwave'], n_disasters
        )
        
        # Standardize
        x_value = (log_values - scaling_params['log_value_mean']) / scaling_params['log_value_std']
        x_peak = (peak_intensities - scaling_params['peak_mean']) / scaling_params['peak_std']
        x_duration = (durations - scaling_params['duration_mean']) / scaling_params['duration_std']
        x_percent = (percents - scaling_params['percent_mean']) / scaling_params['percent_std']
        
        # Predict
        log_appropriations = (
            beta_0 + 
            beta_value * x_value +
            beta_peak * x_peak +
            beta_duration * x_duration +
            beta_percent * x_percent +
            np.random.normal(0, sigma, size=n_disasters)
        )
        
        appropriations = np.exp(log_appropriations)
        appropriations = np.clip(appropriations, 0, 2_500_000_000)
        
        results.append({
            'n_disasters': n_disasters,
            'total_appropriation': appropriations.sum()
        })
    
    return pd.DataFrame(results)

# ============================================================================
# Load Data
# ============================================================================

# Load all data
try:
    model_data = load_model_data()
    fitted_distributions = load_fitted_distributions()
    posterior_samples = load_posterior_samples()
    scaling_params = load_scaling_params()
    region_data = load_region_data()
    shapefiles = load_shapefiles()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# ============================================================================
# Dashboard Header
# ============================================================================

st.markdown('<h1 class="main-header">🌊 Fishery Disaster Risk Dashboard</h1>', 
            unsafe_allow_html=True)

st.markdown("""
This interactive dashboard allows you to explore:
- **Spatial data**: Marine heatwaves and fishery locations
- **Historical disasters**: Past events and appropriations
- **Climate scenarios**: Custom predictions with adjustable parameters
- **Monte Carlo simulations**: Probabilistic appropriation forecasts
""")

# ============================================================================
# Sidebar: Navigation & Controls
# ============================================================================

st.sidebar.title("🎛️ Dashboard Controls")

page = st.sidebar.radio(
    "Navigate to:",
    ["📍 Spatial Explorer", "📜 Historical Disasters", "🌡️ Climate Scenarios", "📊 Distributions"]
)

st.sidebar.markdown("---")

# ============================================================================
# PAGE 1: SPATIAL EXPLORER
# ============================================================================

if page == "📍 Spatial Explorer":
    st.markdown('<h2 class="sub-header">📍 Spatial & Temporal Heatwave Explorer</h2>', 
                unsafe_allow_html=True)
    
    if not region_data:
        st.warning("⚠️ No spatial data found. Please ensure data files are in the correct locations.")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("🎨 Controls")
            
            # Region/Species selection
            st.markdown("**Select Region/Species:**")
            all_options = sorted(list(region_data.keys()))
            
            selected_region = st.selectbox(
                "Choose one to visualize:",
                all_options,
                help="Select a region or species to explore heatwave events"
            )
            
            if selected_region:
                info = region_data[selected_region]
                
                st.markdown("---")
                st.markdown("**Data Available:**")
                if info['has_mask']:
                    st.success("✅ Spatial Mask")
                if info['has_output']:
                    st.success("✅ Filtered Events (✨ Using this)")
                elif info['has_sst']:
                    st.warning("⚠️ Only raw events available")
                    st.info("💡 Will show events outside region boundary")
                else:
                    st.error("❌ No heatwave data found")
                
                # Load heatwave data to get date range
                hw_df = None
                if info['has_output']:
                    # Prioritize filtered output - this is already masked to the region!
                    hw_df = load_heatwave_csv(info['paths']['csv_output_path'])
                elif info['has_sst']:
                    # Fall back to raw SST data (will contain ALL events for parent region)
                    hw_df = load_heatwave_csv(info['paths']['sst_data_path'])
                    st.warning("⚠️ Using unfiltered data - events may be outside selected region")
                
                # Show available columns for debugging
                if hw_df is not None:
                    with st.expander("🔍 Debug: View Data Columns"):
                        st.code(", ".join(hw_df.columns.tolist()))
                        st.caption(f"Total columns: {len(hw_df.columns)}")
                        st.caption(f"Total rows: {len(hw_df)}")
                
                if hw_df is not None:
                    # Parse dates - try multiple possible column names
                    date_start_col = None
                    date_end_col = None
                    
                    # Try different possible column name patterns
                    start_patterns = ['start_date', 'time_start', 'date_start', 'start', 'date_peak']
                    end_patterns = ['end_date', 'time_end', 'date_end', 'end', 'date_peak']
                    
                    for pattern in start_patterns:
                        if pattern in hw_df.columns:
                            date_start_col = pattern
                            break
                    
                    for pattern in end_patterns:
                        if pattern in hw_df.columns:
                            date_end_col = pattern
                            break
                    
                    # If we found date columns, parse them
                    if date_start_col and date_end_col:
                        try:
                            hw_df[date_start_col] = pd.to_datetime(hw_df[date_start_col])
                            hw_df[date_end_col] = pd.to_datetime(hw_df[date_end_col])
                            
                            min_date = hw_df[date_start_col].min()
                            max_date = hw_df[date_end_col].max()
                            
                            st.markdown("---")
                            st.markdown("**📅 Select Date:**")
                            
                            # Initialize session state for persistent date
                            if 'spatial_selected_date' not in st.session_state:
                                st.session_state.spatial_selected_date = min_date.date()
                            
                            # Ensure selected date is within data range
                            if st.session_state.spatial_selected_date < min_date.date():
                                st.session_state.spatial_selected_date = min_date.date()
                            elif st.session_state.spatial_selected_date > max_date.date():
                                st.session_state.spatial_selected_date = max_date.date()
                            
                            # Simple date picker
                            selected_date = st.date_input(
                                "Choose a date:",
                                value=st.session_state.spatial_selected_date,
                                min_value=min_date.date(),
                                max_value=max_date.date(),
                                help="Date persists when switching regions",
                                key=f"date_picker_{selected_region}"
                            )
                            
                            # Update session state
                            st.session_state.spatial_selected_date = selected_date
                            selected_datetime = pd.Timestamp(selected_date)
                            
                            # Filter events active on selected date
                            active_events = hw_df[
                                (hw_df[date_start_col] <= selected_datetime) & 
                                (hw_df[date_end_col] >= selected_datetime)
                            ]
                            
                            st.markdown("---")
                            st.markdown(f"**📊 Active on {selected_date}:**")
                            st.metric("Heatwave Events", len(active_events))
                            
                            if len(active_events) > 0:
                                st.metric("Avg Intensity", 
                                         f"{active_events['intensity_max'].mean():.2f}°C" 
                                         if 'intensity_max' in active_events.columns else "N/A")
                        
                        except Exception as e:
                            st.warning(f"⚠️ Could not parse dates: {e}")
                            st.info(f"Found columns: {date_start_col}, {date_end_col}")
                            selected_datetime = None
                            active_events = hw_df
                            selected_date = "All Time"
                    
                    else:
                        # No date columns found - show available columns for debugging
                        st.warning("⚠️ No date columns found in heatwave data")
                        st.info(f"Available columns: {', '.join(hw_df.columns[:10])}")
                        
                        # Just use all events
                        selected_datetime = None
                        active_events = hw_df
                        selected_date = "All Time"
                        
                        st.markdown("---")
                        st.markdown("**Showing All Events:**")
                        st.metric("Total Events", len(active_events))
                    
                    # Look for individual event files
                    st.markdown("---")
                    st.markdown("**Individual Events:**")
                    
                    # Get directory of filtered events
                    output_path = Path(info['paths']['csv_output_path'])
                    event_dir = output_path.parent
                    base_name = selected_region.lower()
                    
                    # Find event files matching pattern
                    event_files = []
                    if event_dir.exists():
                        for pattern in [
                            f"*{base_name}*event_*.csv",
                            f"*event_*.csv"
                        ]:
                            event_files.extend(list(event_dir.glob(pattern)))
                    
                    if event_files:
                        st.success(f"Found {len(event_files)} individual event files")
                        
                        # Select specific event
                        event_names = [f.stem for f in event_files]
                        selected_event_file = st.selectbox(
                            "View specific event:",
                            ["All Events"] + event_names
                        )
                    else:
                        st.info("No individual event files found")
                        selected_event_file = "All Events"
                else:
                    st.warning("⚠️ No heatwave data available")
                    selected_datetime = None
                    active_events = None
                    selected_event_file = "All Events"
        
        with col2:
            if not selected_region:
                st.info("👈 Select a region/species from the sidebar")
            else:
                info = region_data[selected_region]
                
                # SECTION 1: SPATIAL MAP
                st.subheader(f"🗺️ {selected_region.replace('_', ' ')} - Spatial Extent")
                
                if info['has_mask'] and HAS_XARRAY:
                    try:
                        mask_data = load_spatial_mask(info['paths']['mask'])
                        
                        if mask_data is not None and 'error' not in mask_data:
                            lats = mask_data['lats']
                            lons = mask_data['lons']
                            mask = mask_data['mask']
                            
                            st.success(f"✅ Loaded mask: '{mask_data['mask_var']}' using coordinates ({mask_data['lat_coord']}, {mask_data['lon_coord']})")
                            st.info(f"📊 Cells inside region: {mask_data['cells_inside']:,}")
                            
                            # Handle 3D masks (take first slice if needed)
                            if len(mask.shape) == 3:
                                st.info(f"📊 Mask is 3D (shape: {mask.shape}), using first slice")
                                mask = mask[0]
                            elif len(mask.shape) > 3:
                                st.info(f"📊 Mask is {len(mask.shape)}D (shape: {mask.shape}), reducing to 2D")
                                while len(mask.shape) > 2:
                                    mask = mask[0]
                            
                            # Inspect mask values
                            st.info(f"📊 Mask shape: {mask.shape}, NaN cells: {np.sum(np.isnan(mask)):,}, Non-NaN cells: {np.sum(~np.isnan(mask)):,}")
                            
                            # Use the same logic as user's code: non-NaN = inside region
                            mask_binary = ~np.isnan(mask)
                            
                            n_cells = np.sum(mask_binary)
                            if n_cells == 0:
                                st.error(f"❌ All mask values are NaN!")
                                st.write("**Mask Statistics:**")
                                st.write(f"- Shape: {mask.shape}")
                                st.write(f"- All values are NaN: {np.all(np.isnan(mask))}")
                                st.write(f"- Sample values: {mask.flatten()[:10]}")
                            else:
                                st.success(f"✓ Valid mask with {n_cells:,} cells inside region")
                                
                                # Get bounding box for reference
                                lat_indices, lon_indices = np.where(mask_binary)
                                if len(lat_indices) > 0:
                                    lat_min, lat_max = lats[lat_indices].min(), lats[lat_indices].max()
                                    lon_min, lon_max = lons[lon_indices].min(), lons[lon_indices].max()
                                    
                                    st.info(f"📐 Region bounds: Lat [{lat_min:.2f}, {lat_max:.2f}], Lon [{lon_min:.2f}, {lon_max:.2f}]")
                                    
                                    # Use FIXED map extent for consistent sizing
                                    # Covers Alaska to California, Pacific coast
                                    MAP_LAT_MIN = 25  # Southern California
                                    MAP_LAT_MAX = 70  # Northern Alaska
                                    MAP_LON_MIN = -180  # Western Alaska/Pacific
                                    MAP_LON_MAX = -110  # Western US border
                                    
                                    # Create focused map
                                    fig = go.Figure()
                                    
                                    # Plot spatial mask
                                    mask_lons, mask_lats = np.meshgrid(lons, lats)
                                    
                                    fig.add_trace(go.Scattergeo(
                                        lon=mask_lons[mask_binary].flatten(),
                                        lat=mask_lats[mask_binary].flatten(),
                                        mode='markers',
                                        marker=dict(
                                            size=3,
                                            color='steelblue',
                                            opacity=0.4
                                        ),
                                        name='Fishery Extent',
                                        hoverinfo='skip'
                                    ))
                                    
                                    # Overlay active heatwave events
                                    if active_events is not None and len(active_events) > 0:
                                        # Check if spatial data exists in events - try multiple column names
                                        lat_col = None
                                        lon_col = None
                                        
                                        for col in ['latitude', 'lat', 'Latitude', 'LAT']:
                                            if col in active_events.columns:
                                                lat_col = col
                                                break
                                        
                                        for col in ['longitude', 'lon', 'Longitude', 'LON']:
                                            if col in active_events.columns:
                                                lon_col = col
                                                break
                                        
                                        if lat_col and lon_col:
                                            # Plot individual event locations
                                            fig.add_trace(go.Scattergeo(
                                                lon=active_events[lon_col],
                                                lat=active_events[lat_col],
                                                mode='markers',
                                                marker=dict(
                                                    size=8,
                                                    color=active_events['intensity_max'] if 'intensity_max' in active_events.columns else 'red',
                                                    colorscale='Reds',
                                                    colorbar=dict(title="Intensity (°C)"),
                                                    line=dict(width=1, color='darkred')
                                                ),
                                                text=active_events.apply(lambda x: 
                                                    f"Event ID: {x.name}<br>" +
                                                    f"Intensity: {x.get('intensity_max', 'N/A'):.2f}°C<br>" +
                                                    f"Duration: {x.get('duration', 'N/A'):.0f} days"
                                                    if 'intensity_max' in active_events.columns else f"Event {x.name}", axis=1),
                                                hovertemplate='%{text}<extra></extra>',
                                                name='Active Heatwaves'
                                            ))
                                        else:
                                            st.info(f"📍 No location data in events (looking for latitude/longitude columns)")
                                    
                                    # Load specific event file if selected
                                    if selected_event_file != "All Events":
                                        event_file_path = event_dir / f"{selected_event_file}.csv"
                                        if event_file_path.exists():
                                            event_df = pd.read_csv(event_file_path)
                                            
                                            # Check for lat/lon columns
                                            lat_col = None
                                            lon_col = None
                                            
                                            for col in ['latitude', 'lat', 'Latitude', 'LAT']:
                                                if col in event_df.columns:
                                                    lat_col = col
                                                    break
                                            
                                            for col in ['longitude', 'lon', 'Longitude', 'LON']:
                                                if col in event_df.columns:
                                                    lon_col = col
                                                    break
                                            
                                            if lat_col and lon_col:
                                                fig.add_trace(go.Scattergeo(
                                                    lon=event_df[lon_col],
                                                    lat=event_df[lat_col],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=10,
                                                        color='orange',
                                                        symbol='star',
                                                        line=dict(width=2, color='darkorange')
                                                    ),
                                                    name=f'Event {selected_event_file}',
                                                    hovertemplate='Specific Event<extra></extra>'
                                                ))
                                    
                                    # Update layout for fixed view
                                    fig.update_geos(
                                        projection_type="mercator",
                                        showcountries=True,
                                        showcoastlines=True,
                                        showland=True,
                                        landcolor="rgb(243, 243, 243)",
                                        coastlinecolor="rgb(100, 100, 100)",
                                        showocean=True,
                                        oceancolor="rgb(230, 245, 255)",
                                        lataxis_range=[MAP_LAT_MIN, MAP_LAT_MAX],
                                        lonaxis_range=[MAP_LON_MIN, MAP_LON_MAX],
                                        resolution=50
                                    )
                                    
                                    fig.update_layout(
                                        height=800,  # Increased from 600
                                        title_text=f"{selected_region.replace('_', ' ')} - {selected_date if selected_datetime else 'All Time'}",
                                        showlegend=True,
                                        margin=dict(l=0, r=0, t=50, b=0)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show event details
                                    if active_events is not None and len(active_events) > 0:
                                        st.markdown("---")
                                        st.subheader(f"📊 Active Events on {selected_date}")
                                        
                                        # Summary stats
                                        col_a, col_b, col_c, col_d = st.columns(4)
                                        
                                        with col_a:
                                            st.metric("Events", len(active_events))
                                        
                                        with col_b:
                                            if 'intensity_max' in active_events.columns:
                                                st.metric("Max Intensity", 
                                                         f"{active_events['intensity_max'].max():.2f}°C")
                                        
                                        with col_c:
                                            if 'intensity_mean' in active_events.columns:
                                                st.metric("Avg Intensity", 
                                                         f"{active_events['intensity_mean'].mean():.2f}°C")
                                        
                                        with col_d:
                                            if 'duration' in active_events.columns:
                                                st.metric("Avg Duration", 
                                                         f"{active_events['duration'].mean():.0f} days")
                                        
                                        # Data table
                                        with st.expander("📋 View Event Data"):
                                            display_cols = [col for col in active_events.columns if col in [
                                                'start_date', 'end_date', 'time_start', 'time_end', 
                                                'duration', 'intensity_max', 'intensity_mean', 
                                                'intensity_cumulative', 'latitude', 'longitude', 'lat', 'lon'
                                            ]]
                                            
                                            if display_cols:
                                                st.dataframe(
                                                    active_events[display_cols],
                                                    use_container_width=True
                                                )
                                            else:
                                                st.dataframe(active_events, use_container_width=True)
                                    
                                    # Show specific event details if selected
                                    if selected_event_file != "All Events":
                                        st.markdown("---")
                                        st.subheader(f"🎯 Individual Event: {selected_event_file}")
                                        
                                        event_file_path = event_dir / f"{selected_event_file}.csv"
                                        if event_file_path.exists():
                                            event_df = pd.read_csv(event_file_path)
                                            
                                            col_e, col_f, col_g = st.columns(3)
                                            
                                            with col_e:
                                                st.metric("Data Points", len(event_df))
                                            
                                            with col_f:
                                                if 'sst' in event_df.columns:
                                                    st.metric("Avg SST", f"{event_df['sst'].mean():.2f}°C")
                                            
                                            with col_g:
                                                if 'sst_anom' in event_df.columns:
                                                    st.metric("Avg Anomaly", f"{event_df['sst_anom'].mean():.2f}°C")
                                            
                                            with st.expander("📋 View Full Event Data"):
                                                st.dataframe(event_df.head(100), use_container_width=True)
                                
                                else:
                                    st.warning("⚠️ No lat/lon indices found in mask")
                        
                        elif mask_data is not None and 'error' in mask_data:
                            # Show detailed error information
                            st.error(f"❌ Could not load spatial mask: {mask_data['error']}")
                            
                            with st.expander("🔍 Debug Information"):
                                if 'available_vars' in mask_data:
                                    st.write("**Available variables in NetCDF:**")
                                    st.code(", ".join(mask_data['available_vars']))
                                
                                if 'available_coords' in mask_data:
                                    st.write("**Available coordinates:**")
                                    st.code(", ".join(mask_data['available_coords']))
                                
                                if 'available_dims' in mask_data:
                                    st.write("**Available dimensions:**")
                                    st.code(", ".join(mask_data['available_dims']))
                                
                                if 'mask_var' in mask_data:
                                    st.write(f"**Found mask variable:** {mask_data['mask_var']}")
                                
                                if 'traceback' in mask_data:
                                    st.write("**Full error:**")
                                    st.code(mask_data['traceback'])
                                
                                st.write(f"**File path:** {info['paths']['mask']}")
                        
                        else:
                            st.warning("⚠️ Could not load spatial mask")
                    
                    except Exception as e:
                        st.error(f"Error creating map: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                
                elif not HAS_XARRAY:
                    st.warning("⚠️ Install xarray and netCDF4 to view spatial maps: `pip install xarray netCDF4`")
                else:
                    st.info("No spatial mask file available for this region")

# ============================================================================
# PAGE 2: HISTORICAL DISASTERS
# ============================================================================

elif page == "📜 Historical Disasters":
    st.markdown('<h2 class="sub-header">📜 Historical Disaster Explorer</h2>', 
                unsafe_allow_html=True)
    
    if not data_loaded:
        st.error("❌ Could not load historical disaster data")
    else:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Disasters",
                len(model_data),
                help="Number of historical disasters in dataset"
            )
        
        with col2:
            st.metric(
                "Total Appropriated",
                f"${model_data['total_appropriation'].sum()/1e9:.2f}B",
                help="Total federal appropriations (1992-2023)"
            )
        
        with col3:
            st.metric(
                "Avg per Disaster",
                f"${model_data['total_appropriation'].mean()/1e6:.1f}M",
                help="Mean appropriation per disaster"
            )
        
        with col4:
            disasters_per_year = model_data.groupby('sst_year').size()
            st.metric(
                "Avg per Year",
                f"{disasters_per_year.mean():.1f}",
                help="Average disasters per year"
            )
        
        st.markdown("---")
        
        # Filters
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔍 Filters")
            
            # Year filter
            year_filter = st.slider(
                "Year Range:",
                int(model_data['sst_year'].min()),
                int(model_data['sst_year'].max()),
                (int(model_data['sst_year'].min()), int(model_data['sst_year'].max()))
            )
            
            # Region filter
            if 'region' in model_data.columns:
                regions = ['All'] + sorted(model_data['region'].unique().tolist())
                region_filter = st.selectbox("Region:", regions)
            else:
                region_filter = 'All'
            
            # Appropriation filter
            min_approp = st.number_input(
                "Min Appropriation ($M):",
                min_value=0,
                max_value=int(model_data['total_appropriation'].max()/1e6),
                value=0
            )
        
        # Apply filters
        filtered_data = model_data[
            (model_data['sst_year'] >= year_filter[0]) &
            (model_data['sst_year'] <= year_filter[1]) &
            (model_data['total_appropriation'] >= min_approp * 1e6)
        ]
        
        if region_filter != 'All' and 'region' in model_data.columns:
            filtered_data = filtered_data[filtered_data['region'] == region_filter]
        
        with col2:
            st.subheader(f"📊 Disasters ({len(filtered_data)} shown)")
            
            # Timeline plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=filtered_data['sst_year'],
                y=filtered_data['total_appropriation'] / 1e6,
                mode='markers',
                marker=dict(
                    size=10,
                    color=filtered_data['peak_intensity'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Peak<br>Intensity<br>(°C)")
                ),
                text=filtered_data.apply(lambda x: 
                    f"Year: {x['sst_year']}<br>" +
                    f"Appropriation: ${x['total_appropriation']/1e6:.1f}M<br>" +
                    f"Peak Intensity: {x['peak_intensity']:.1f}°C<br>" +
                    f"Duration: {x['duration_days']/1e6:.2f}M days", axis=1),
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Historical Disasters Timeline",
                xaxis_title="Year",
                yaxis_title="Appropriation ($ Millions)",
                height=400,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Disaster Details")
        
        display_cols = ['sst_year', 'region', 'total_appropriation', 
                       'peak_intensity', 'duration_days', 'percent_in_heatwave']
        display_cols = [col for col in display_cols if col in filtered_data.columns]
        
        display_data = filtered_data[display_cols].copy()
        display_data['total_appropriation'] = display_data['total_appropriation'].apply(
            lambda x: f"${x/1e6:.2f}M"
        )
        display_data['duration_days'] = display_data['duration_days'].apply(
            lambda x: f"{x/1e6:.3f}M"
        )
        
        st.dataframe(
            display_data.sort_values('sst_year', ascending=False),
            use_container_width=True,
            height=400
        )

# ============================================================================
# PAGE 3: CLIMATE SCENARIOS
# ============================================================================

elif page == "🌡️ Climate Scenarios":
    st.markdown('<h2 class="sub-header">🌡️ Climate Scenario Comparison</h2>', 
                unsafe_allow_html=True)
    
    if not data_loaded:
        st.error("❌ Could not load model data")
    else:
        st.markdown("""
        Compare baseline historical conditions with custom climate scenarios.
        Adjust sliders to see how changes affect predicted appropriations.
        """)
        
        # Controls sidebar
        with st.sidebar:
            st.markdown("### 🎛️ Scenario Parameters")
            
            # Frequency
            frequency_mult = st.slider(
                "🔢 Disaster Frequency",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                format="%.1fx",
                help="Multiplier for number of disasters per year"
            )
            freq_pct = (frequency_mult - 1.0) * 100
            st.caption(f"Change: {freq_pct:+.0f}%")
            
            # Intensity
            intensity_mult = st.slider(
                "🌡️ Peak Intensity",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                format="%.1fx",
                help="Multiplier for heatwave peak intensity"
            )
            int_pct = (intensity_mult - 1.0) * 100
            st.caption(f"Change: {int_pct:+.0f}%")
            
            # Duration
            duration_mult = st.slider(
                "⏱️ Duration",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                format="%.1fx",
                help="Multiplier for heatwave duration"
            )
            dur_pct = (duration_mult - 1.0) * 100
            st.caption(f"Change: {dur_pct:+.0f}%")
            
            # Coverage
            coverage_mult = st.slider(
                "📐 Spatial Coverage",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.05,
                format="%.2fx",
                help="Multiplier for spatial coverage"
            )
            cov_pct = (coverage_mult - 1.0) * 100
            st.caption(f"Change: {cov_pct:+.0f}%")
            
            st.markdown("---")
            
            # Simulation parameters
            n_sims = st.select_slider(
                "🎲 Simulations",
                options=[100, 500, 1000, 2000, 5000],
                value=1000,
                help="More simulations = more accurate but slower"
            )
            
            # Run button
            run_simulation = st.button("▶️ Run Both Scenarios", type="primary", use_container_width=True)
        
        # Two-pane comparison
        col_baseline, col_custom = st.columns(2)
        
        with col_baseline:
            st.subheader("📈 Baseline (Historical)")
            st.info("Historical conditions with no climate change")
            
            # Always show baseline
            if 'baseline_results' not in st.session_state or run_simulation:
                with st.spinner(f"Running baseline ({n_sims:,} simulations)..."):
                    baseline_results = run_monte_carlo_simulation(
                        fitted_distributions,
                        posterior_samples,
                        scaling_params,
                        frequency_multiplier=1.0,
                        intensity_multiplier=1.0,
                        duration_multiplier=1.0,
                        coverage_multiplier=1.0,
                        n_simulations=n_sims
                    )
                    st.session_state.baseline_results = baseline_results
            
            baseline_results = st.session_state.baseline_results
            
            # Display baseline results with full graphs
            mean_approp = baseline_results['total_appropriation'].mean()
            median_approp = baseline_results['total_appropriation'].median()
            ci_low = baseline_results['total_appropriation'].quantile(0.025)
            ci_high = baseline_results['total_appropriation'].quantile(0.975)
            prob_100m = (baseline_results['total_appropriation'] > 100e6).mean()
            prob_200m = (baseline_results['total_appropriation'] > 200e6).mean()
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Mean", f"${mean_approp/1e6:.1f}M")
            with col_b:
                st.metric("Median", f"${median_approp/1e6:.1f}M")
            with col_c:
                st.metric("95% CI", f"${ci_low/1e6:.0f}M - ${ci_high/1e6:.0f}M")
            
            col_d, col_e = st.columns(2)
            with col_d:
                st.metric("P(>$100M)", f"{prob_100m:.1%}")
            with col_e:
                st.metric("P(>$200M)", f"{prob_200m:.1%}")
            
            # Distribution histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=baseline_results['total_appropriation'] / 1e6,
                nbinsx=50,
                marker_color='steelblue',
                opacity=0.7,
                name='Baseline'
            ))
            fig.add_vline(x=mean_approp/1e6, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: ${mean_approp/1e6:.1f}M")
            fig.add_vline(x=median_approp/1e6, line_dash="dot", line_color="orange",
                         annotation_text=f"Median: ${median_approp/1e6:.1f}M")
            fig.update_layout(
                title="Baseline Distribution",
                xaxis_title="Appropriation ($M)",
                yaxis_title="Frequency",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key="baseline_distribution")
            
            # Disaster frequency chart
            disaster_counts = baseline_results['n_disasters'].value_counts().sort_index()
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=disaster_counts.index,
                y=disaster_counts.values,
                marker_color='steelblue',
                opacity=0.7
            ))
            fig2.update_layout(
                title="Baseline Disaster Frequency",
                xaxis_title="Number of Disasters",
                yaxis_title="Frequency",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True, key="baseline_frequency")
        
        with col_custom:
            st.subheader("🌡️ Custom Scenario")
            st.info("Adjust sliders and click Run to see predictions")
            
            if run_simulation:
                with st.spinner(f"Running custom scenario ({n_sims:,} simulations)..."):
                    results = run_monte_carlo_simulation(
                        fitted_distributions,
                        posterior_samples,
                        scaling_params,
                        frequency_multiplier=frequency_mult,
                        intensity_multiplier=intensity_mult,
                        duration_multiplier=duration_mult,
                        coverage_multiplier=coverage_mult,
                        n_simulations=n_sims
                    )
                
                # Store in session state
                st.session_state['mc_results'] = results
                st.session_state['scenario_params'] = {
                    'frequency': frequency_mult,
                    'intensity': intensity_mult,
                    'duration': duration_mult,
                    'coverage': coverage_mult
                }
            
            # Display results if available
            if 'mc_results' in st.session_state:
                results = st.session_state['mc_results']
                
                # Summary metrics
                mean_approp = results['total_appropriation'].mean()
                median_approp = results['total_appropriation'].median()
                ci_low = results['total_appropriation'].quantile(0.025)
                ci_high = results['total_appropriation'].quantile(0.975)
                prob_100m = (results['total_appropriation'] > 100e6).mean()
                prob_200m = (results['total_appropriation'] > 200e6).mean()
                
                # Calculate deltas from baseline
                if 'baseline_results' in st.session_state:
                    baseline = st.session_state.baseline_results
                    baseline_mean = baseline['total_appropriation'].mean()
                    baseline_median = baseline['total_appropriation'].median()
                    baseline_prob_100m = (baseline['total_appropriation'] > 100e6).mean()
                    baseline_prob_200m = (baseline['total_appropriation'] > 200e6).mean()
                    
                    delta_mean = ((mean_approp - baseline_mean) / baseline_mean) * 100
                    delta_median = ((median_approp - baseline_median) / baseline_median) * 100
                    delta_prob_100m = (prob_100m - baseline_prob_100m) * 100  # percentage points
                    delta_prob_200m = (prob_200m - baseline_prob_200m) * 100
                else:
                    delta_mean = None
                    delta_median = None
                    delta_prob_100m = None
                    delta_prob_200m = None
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Mean",
                        f"${mean_approp/1e6:.1f}M",
                        delta=f"{delta_mean:+.1f}%" if delta_mean is not None else None,
                        help="Average yearly appropriation"
                    )
                
                with col_b:
                    st.metric(
                        "Median",
                        f"${median_approp/1e6:.1f}M",
                        delta=f"{delta_median:+.1f}%" if delta_median is not None else None,
                        help="50th percentile"
                    )
                
                with col_c:
                    st.metric(
                        "95% CI",
                        f"${ci_low/1e6:.0f}M - ${ci_high/1e6:.0f}M",
                        help="95% confidence interval"
                    )
                
                col_d, col_e = st.columns(2)
                
                with col_d:
                    st.metric(
                        "P(>$100M)",
                        f"{prob_100m:.1%}",
                        delta=f"{delta_prob_100m:+.1f} pts" if delta_prob_100m is not None else None,
                        help="Probability of exceeding $100M"
                    )
                
                with col_e:
                    st.metric(
                        "P(>$200M)",
                        f"{prob_200m:.1%}",
                        delta=f"{delta_prob_200m:+.1f} pts" if delta_prob_200m is not None else None,
                        help="Probability of exceeding $200M"
                    )
                
                # Distribution histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=results['total_appropriation'] / 1e6,
                    nbinsx=50,
                    marker_color='orange',
                    opacity=0.7,
                    name='Custom Scenario'
                ))
                fig.add_vline(x=mean_approp/1e6, line_dash="dash", line_color="red",
                             annotation_text=f"Mean: ${mean_approp/1e6:.1f}M")
                fig.add_vline(x=median_approp/1e6, line_dash="dot", line_color="darkred",
                             annotation_text=f"Median: ${median_approp/1e6:.1f}M")
                fig.update_layout(
                    title="Custom Scenario Distribution",
                    xaxis_title="Appropriation ($M)",
                    yaxis_title="Frequency",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, key="custom_distribution")
                
                # Disaster frequency chart
                disaster_counts = results['n_disasters'].value_counts().sort_index()
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=disaster_counts.index,
                    y=disaster_counts.values,
                    marker_color='orange',
                    opacity=0.7
                ))
                fig2.update_layout(
                    title="Custom Scenario Disaster Frequency",
                    xaxis_title="Number of Disasters",
                    yaxis_title="Frequency",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True, key="custom_frequency")
            
            else:
                st.info("👆 Click 'Run Both Scenarios' to generate predictions")

# ============================================================================
# PAGE 4: DISTRIBUTIONS
# ============================================================================

elif page == "📊 Distributions":
    st.markdown('<h2 class="sub-header">📊 Distribution Explorer</h2>', 
                unsafe_allow_html=True)
    
    if not data_loaded:
        st.error("❌ Could not load distribution data")
    else:
        st.markdown("""
        Explore the fitted probability distributions for all predictor variables 
        and simulated appropriations.
        """)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Predictor Distributions", "💰 Appropriation Predictions", "📋 Summary Statistics"])
        
        with tab1:
            st.subheader("Fitted Distributions for Heatwave Characteristics")
            
            # Create 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Peak Intensity
            ax = axes[0, 0]
            data = model_data['peak_intensity']
            ax.hist(data, bins=30, density=True, alpha=0.6, color='steelblue', 
                   edgecolor='black', label='Historical')
            
            if 'peak_intensity' in fitted_distributions:
                dist_info = fitted_distributions['peak_intensity']
                dist = getattr(stats, dist_info['distribution'])
                x = np.linspace(data.min(), data.max(), 100)
                y = dist.pdf(x, *dist_info['params'])
                ax.plot(x, y, 'r-', linewidth=2, label=f"Fitted {dist_info['distribution']}")
            
            ax.set_xlabel('Peak Intensity (°C)')
            ax.set_ylabel('Density')
            ax.set_title('Peak Heatwave Intensity')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Duration
            ax = axes[0, 1]
            data = model_data['duration_days'] / 1e6
            ax.hist(data, bins=30, density=True, alpha=0.6, color='steelblue',
                   edgecolor='black', label='Historical')
            
            if 'duration_days' in fitted_distributions:
                dist_info = fitted_distributions['duration_days']
                dist = getattr(stats, dist_info['distribution'])
                x = np.linspace(0, data.max(), 100)
                y = dist.pdf(x * 1e6, *dist_info['params']) * 1e6
                ax.plot(x, y, 'r-', linewidth=2, label=f"Fitted {dist_info['distribution']}")
            
            ax.set_xlabel('Duration (Million Days)')
            ax.set_ylabel('Density')
            ax.set_title('Heatwave Duration')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Spatial Coverage
            ax = axes[1, 0]
            data = model_data['percent_in_heatwave']
            ax.hist(data, bins=30, density=True, alpha=0.6, color='steelblue',
                   edgecolor='black', label='Historical')
            
            if 'percent_in_heatwave' in fitted_distributions:
                dist_info = fitted_distributions['percent_in_heatwave']
                dist = getattr(stats, dist_info['distribution'])
                x = np.linspace(0, 100, 100)
                if 'scale' in dist_info:
                    y = dist.pdf(x / dist_info['scale'], *dist_info['params']) / dist_info['scale']
                else:
                    y = dist.pdf(x, *dist_info['params'])
                ax.plot(x, y, 'r-', linewidth=2, label=f"Fitted {dist_info['distribution']}")
            
            ax.set_xlabel('Spatial Coverage (%)')
            ax.set_ylabel('Density')
            ax.set_title('Heatwave Spatial Coverage')
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Disaster Frequency
            ax = axes[1, 1]
            disasters_per_year = model_data.groupby('sst_year').size()
            counts = disasters_per_year.value_counts().sort_index()
            ax.bar(counts.index, counts.values / len(disasters_per_year), 
                  alpha=0.6, color='steelblue', edgecolor='black', label='Historical')
            
            if 'disaster_frequency' in fitted_distributions:
                dist_info = fitted_distributions['disaster_frequency']
                if dist_info['distribution'] == 'zt_poisson':
                    lam = dist_info['params']['lambda']
                    x = np.arange(1, 10)
                    y = [(lam**k / math.factorial(k)) / (1 - np.exp(-lam)) for k in x]
                    ax.plot(x, y, 'ro-', linewidth=2, markersize=8, label='Fitted ZT-Poisson')
            
            ax.set_xlabel('Disasters per Year')
            ax.set_ylabel('Probability')
            ax.set_title('Disaster Frequency')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Appropriation Distribution")
            
            # Show current scenario results if available
            if 'mc_results' in st.session_state:
                results = st.session_state['mc_results']
                params = st.session_state['scenario_params']
                
                st.markdown(f"""
                **Current Scenario:**
                - Frequency: {params['frequency']:.1f}x ({(params['frequency']-1)*100:+.0f}%)
                - Intensity: {params['intensity']:.1f}x ({(params['intensity']-1)*100:+.0f}%)
                - Duration: {params['duration']:.1f}x ({(params['duration']-1)*100:+.0f}%)
                - Coverage: {params['coverage']:.2f}x ({(params['coverage']-1)*100:+.0f}%)
                """)
                
                # Create distribution plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Histogram with KDE
                data = results['total_appropriation'] / 1e6
                ax1.hist(data, bins=50, density=True, alpha=0.6, 
                        color='steelblue', edgecolor='black')
                
                # Add KDE
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(0, data.quantile(0.99), 500)
                ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                
                # Add percentiles
                ax1.axvline(data.quantile(0.25), color='green', linestyle='--', 
                           alpha=0.7, label='25th %ile')
                ax1.axvline(data.quantile(0.50), color='orange', linestyle='--',
                           alpha=0.7, label='Median')
                ax1.axvline(data.quantile(0.75), color='red', linestyle='--',
                           alpha=0.7, label='75th %ile')
                
                ax1.set_xlabel('Total Appropriation ($ Millions)')
                ax1.set_ylabel('Density')
                ax1.set_title('Appropriation Distribution')
                ax1.legend()
                ax1.grid(alpha=0.3)
                ax1.set_xlim(0, data.quantile(0.99))
                
                # CDF
                sorted_data = np.sort(data)
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax2.plot(sorted_data, cdf, linewidth=2, color='steelblue')
                
                # Add reference lines
                for prob, label in [(0.25, '25%'), (0.5, '50%'), (0.75, '75%'), (0.95, '95%')]:
                    value = data.quantile(prob)
                    ax2.axhline(prob, color='gray', linestyle=':', alpha=0.5)
                    ax2.axvline(value, color='gray', linestyle=':', alpha=0.5)
                    ax2.plot(value, prob, 'ro', markersize=8)
                    ax2.text(value, prob + 0.02, f'{label}: ${value:.0f}M', 
                            fontsize=8, ha='center')
                
                ax2.set_xlabel('Total Appropriation ($ Millions)')
                ax2.set_ylabel('Cumulative Probability')
                ax2.set_title('Cumulative Distribution Function')
                ax2.grid(alpha=0.3)
                ax2.set_xlim(0, data.quantile(0.99))
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Exceedance probabilities
                st.markdown("**Exceedance Probabilities:**")
                thresholds = [50, 100, 150, 200, 250, 300, 400, 500]
                probs = [(results['total_appropriation'] > t*1e6).mean() for t in thresholds]
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=thresholds,
                    y=probs,
                    mode='lines+markers',
                    line=dict(width=3, color='steelblue'),
                    marker=dict(size=10)
                ))
                
                fig3.update_layout(
                    title="Probability of Exceeding Threshold",
                    xaxis_title="Threshold ($ Millions)",
                    yaxis_title="Probability",
                    height=400
                )
                
                st.plotly_chart(fig3, use_container_width=True)
            
            else:
                st.info("⚠️ No simulation results available. Go to '🌡️ Climate Scenarios' page and run a simulation first.")
        
        with tab3:
            st.subheader("Summary Statistics")
            
            # Historical statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Historical Data (1992-2023):**")
                
                stats_data = {
                    'Metric': [
                        'Total Disasters',
                        'Disasters per Year (mean)',
                        'Peak Intensity (mean)',
                        'Duration (mean)',
                        'Spatial Coverage (mean)',
                        'Appropriation (mean)',
                        'Appropriation (median)',
                        'Appropriation (total)'
                    ],
                    'Value': [
                        f"{len(model_data)}",
                        f"{model_data.groupby('sst_year').size().mean():.2f}",
                        f"{model_data['peak_intensity'].mean():.2f}°C",
                        f"{model_data['duration_days'].mean()/1e6:.3f}M days",
                        f"{model_data['percent_in_heatwave'].mean():.1f}%",
                        f"${model_data['total_appropriation'].mean()/1e6:.1f}M",
                        f"${model_data['total_appropriation'].median()/1e6:.1f}M",
                        f"${model_data['total_appropriation'].sum()/1e9:.2f}B"
                    ]
                }
                
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
            
            with col2:
                if 'mc_results' in st.session_state:
                    st.markdown("**🎲 Simulation Results:**")
                    results = st.session_state['mc_results']
                    
                    sim_stats = {
                        'Metric': [
                            'Simulations Run',
                            'Mean Appropriation',
                            'Median Appropriation',
                            'Std Deviation',
                            '5th Percentile',
                            '95th Percentile',
                            'P(>$100M)',
                            'P(>$200M)'
                        ],
                        'Value': [
                            f"{len(results):,}",
                            f"${results['total_appropriation'].mean()/1e6:.1f}M",
                            f"${results['total_appropriation'].median()/1e6:.1f}M",
                            f"${results['total_appropriation'].std()/1e6:.1f}M",
                            f"${results['total_appropriation'].quantile(0.05)/1e6:.1f}M",
                            f"${results['total_appropriation'].quantile(0.95)/1e6:.1f}M",
                            f"{(results['total_appropriation'] > 100e6).mean():.1%}",
                            f"{(results['total_appropriation'] > 200e6).mean():.1%}"
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(sim_stats), use_container_width=True, hide_index=True)
                else:
                    st.info("Run a simulation to see predicted statistics")
            
            # Distribution parameters
            st.markdown("---")
            st.markdown("**📐 Fitted Distribution Parameters:**")
            
            dist_params = []
            for var_name, dist_info in fitted_distributions.items():
                if var_name == 'disaster_frequency':
                    dist_params.append({
                        'Variable': 'Disaster Frequency',
                        'Distribution': dist_info['distribution'],
                        'Mean': f"{dist_info['mean']:.3f}",
                        'Variance': f"{dist_info['var']:.3f}"
                    })
                else:
                    dist_params.append({
                        'Variable': var_name.replace('_', ' ').title(),
                        'Distribution': dist_info['distribution'],
                        'Mean': f"{dist_info['mean']:.3f}",
                        'Variance': f"{dist_info.get('var', dist_info.get('std', 'N/A'))}".split('.')[0][:6]
                    })
            
            st.dataframe(pd.DataFrame(dist_params), use_container_width=True, hide_index=True)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Fishery Disaster Risk Dashboard</strong></p>
    <p>Data: NOAA Fisheries | Model: Bayesian Regression with Monte Carlo Simulation</p>
    <p>Built with Streamlit • Python • GeoPandas • Plotly</p>
</div>
""", unsafe_allow_html=True)