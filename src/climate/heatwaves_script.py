import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
from datetime import datetime
import marineHeatWaves as mhw
import pickle
from tqdm import tqdm
import random

# ============================================
# HELPER FUNCTIONS
# ============================================

def identify_valid_locations(nc_files, data_dir):
    """
    Identify which locations have valid (non-NaN) data.
    This scans through quickly to find water pixels.
    
    Returns:
        valid_locations: list of (i, j) tuples for valid locations
        lat: latitude array
        lon: longitude array
    """
    print("\n" + "="*60)
    print("IDENTIFYING VALID LOCATIONS")
    print("="*60)
    
    # Get grid info from first file
    with xr.open_dataset(os.path.join(data_dir, nc_files[0])) as ds:
        sst = ds['sst'].isel(zlev=0)
        lon_mask = (ds['longitude'] >= -180) & (ds['longitude'] <= -130)
        lat_mask = (ds['latitude'] >= 47) & (ds['latitude'] <= 66)
        sst_subset = sst.where(lon_mask & lat_mask, drop=True)
        
        lat = sst_subset['latitude'].values
        lon = sst_subset['longitude'].values
        n_lat, n_lon = len(lat), len(lon)
        
        # Load first time step to check for NaN
        first_sst = sst_subset.isel(time=0).values
    
    print(f"\nGrid size: {n_lat} × {n_lon} = {n_lat * n_lon:,} total locations")
    print("Checking which locations have valid data...")
    
    valid_locations = []
    
    for i in tqdm(range(n_lat), desc="Scanning grid"):
        for j in range(n_lon):
            # Check if this location has any valid data (not all NaN)
            if not np.isnan(first_sst[i, j]):
                valid_locations.append((i, j))
    
    print(f"\n✓ Found {len(valid_locations):,} valid locations ({100*len(valid_locations)/(n_lat*n_lon):.1f}% of grid)")
    print(f"✓ Skipping {n_lat*n_lon - len(valid_locations):,} land/masked locations")
    
    return valid_locations, lat, lon

def get_timeseries_for_location(nc_files, data_dir, lat_idx, lon_idx):
    """
    Extract time series for a single location across all years.
    Only loads data for ONE location, not the whole array.
    """
    sst_ts = []
    time_coords = []
    
    for nc_file in nc_files:
        filepath = os.path.join(data_dir, nc_file)
        
        with xr.open_dataset(filepath) as ds:
            # Select single point
            sst_point = ds['sst'].isel(zlev=0, latitude=lat_idx, longitude=lon_idx)
            
            # Load just this point's data
            sst_values = sst_point.values
            time_values = pd.to_datetime(ds['time'].values)
            
            sst_ts.extend(sst_values)
            time_coords.extend(time_values)
    
    return np.array(sst_ts), np.array(time_coords)

def detect_mhw_at_location(sst_ts, time_coords, clim_period):
    """
    Run marineHeatWaves detection for a single location.
    """
    # Check for sufficient valid data
    valid_mask = ~np.isnan(sst_ts)
    if valid_mask.sum() < 365:  # Need at least 1 year
        return None
    
    sst_clean = sst_ts[valid_mask]
    time_clean = time_coords[valid_mask]
    
    # Convert to ordinal time
    try:
        t_clean = np.array([datetime.toordinal(pd.Timestamp(dt).to_pydatetime()) 
                           for dt in time_clean])
    except:
        return None
    
    # Run marineHeatWaves detection
    try:
        mhws, clim = mhw.detect(
            t_clean, 
            sst_clean,
            climatologyPeriod=clim_period,
            pctile=90
        ) # use the defaults
        
        # Extract metrics
        if mhws['n_events'] > 0:
            result = {
                'n_events': mhws['n_events'],
                'total_days': sum(mhws['duration']),
                'mean_duration': np.mean(mhws['duration']),
                'max_duration': max(mhws['duration']),
                'mean_intensity_max': np.mean(mhws['intensity_max']),
                'max_intensity': max(mhws['intensity_max']),
                'total_cumulative': sum(mhws['intensity_cumulative']),
                'mean_rate_onset': np.mean(mhws['rate_onset']),
                'mean_rate_decline': np.mean(mhws['rate_decline'])
            }
        else:
            result = {
                'n_events': 0,
                'total_days': 0,
                'mean_duration': np.nan,
                'max_duration': 0,
                'mean_intensity_max': np.nan,
                'max_intensity': np.nan,
                'total_cumulative': 0,
                'mean_rate_onset': np.nan,
                'mean_rate_decline': np.nan
            }
        
        return result
        
    except Exception as e:
        print(f"\n  Error at location: {e}")
        return None

# ============================================
# TEST MODE - Process small subset
# ============================================

def test_workflow(data_dir='src/climate/sst_data/alaska', n_test_points=5):
    """
    Test the workflow on a small number of random locations.
    """
    print("\n" + "="*60)
    print("TEST MODE - Processing Random Sample")
    print("="*60)
    
    # Get file list
    nc_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_masked.nc')])
    print(f"\nFound {len(nc_files)} files")
    
    # Identify valid locations
    valid_locations, lat, lon = identify_valid_locations(nc_files, data_dir)
    
    # Randomly sample locations
    if len(valid_locations) < n_test_points:
        n_test_points = len(valid_locations)
    
    test_locations = random.sample(valid_locations, n_test_points)
    
    print(f"\n" + "="*60)
    print(f"TESTING ON {n_test_points} RANDOM LOCATIONS")
    print("="*60)
    
    for idx, (i, j) in enumerate(test_locations):
        print(f"\n--- Location {idx+1}/{n_test_points} ---")
        print(f"Grid indices: ({i}, {j})")
        print(f"Coordinates: {lat[i]:.2f}°N, {lon[j]:.2f}°E")
        
        # Get time series
        print("Loading time series across all years...")
        sst_ts, time_coords = get_timeseries_for_location(nc_files, data_dir, i, j)
        
        print(f"  Time series length: {len(sst_ts)} days")
        print(f"  Date range: {time_coords[0].date()} to {time_coords[-1].date()}")
        print(f"  Valid data points: {(~np.isnan(sst_ts)).sum()} ({100*(~np.isnan(sst_ts)).sum()/len(sst_ts):.1f}%)")
        print(f"  SST range: {np.nanmin(sst_ts):.2f}°C to {np.nanmax(sst_ts):.2f}°C")
        
        # Detect MHWs
        print("Running MHW detection...")
        clim_period = [time_coords[0].year, time_coords[-1].year]
        result = detect_mhw_at_location(sst_ts, time_coords, clim_period)
        
        if result is not None:
            print(f"  ✓ MHW Detection successful!")
            print(f"  Number of events: {result['n_events']}")
            if result['n_events'] > 0:
                print(f"  Total MHW days: {result['total_days']:.0f}")
                print(f"  Mean duration: {result['mean_duration']:.1f} days")
                print(f"  Max intensity: {result['max_intensity']:.2f}°C")
        else:
            print(f"  ✗ MHW Detection failed (insufficient data)")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("\nWorkflow verified. Ready to run on all locations.")
    print(f"Estimated time for full run: {len(valid_locations) * 0.2 / 60:.0f} minutes")

# ============================================
# FULL PROCESSING - All valid locations
# ============================================

def process_all_locations_streaming(data_dir='src/climate/sst_data/alaska'):
    """
    Process ALL valid locations one at a time - memory efficient!
    """
    print("\n" + "="*60)
    print("FULL PROCESSING MODE")
    print("Processing all valid locations (one at a time)")
    print("="*60)
    
    # Get file list
    nc_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_masked.nc')])
    print(f"\nFound {len(nc_files)} files")
    print(f"Years: {nc_files[0].split('_')[1]} to {nc_files[-1].split('_')[1]}")
    
    # Identify valid locations (skips NaN/land)
    valid_locations, lat, lon = identify_valid_locations(nc_files, data_dir)
    
    n_lat, n_lon = len(lat), len(lon)
    
    # Get climatology period
    #with xr.open_dataset(os.path.join(data_dir, nc_files[0])) as ds:
    #    first_year = pd.to_datetime(ds['time'].values[0]).year
    #with xr.open_dataset(os.path.join(data_dir, nc_files[-1])) as ds:
    #    last_year = pd.to_datetime(ds['time'].values[-1]).year
    
    # Hard coding climatology period
    clim_period = [1982, 2011]
    print(f"Climatology period: {clim_period[0]}-{clim_period[1]}")
    
    # Initialize result arrays
    print("\nInitializing result arrays...")
    results = {
        'n_events': np.full((n_lat, n_lon), np.nan),
        'total_days': np.full((n_lat, n_lon), np.nan),
        'mean_duration': np.full((n_lat, n_lon), np.nan),
        'max_duration': np.full((n_lat, n_lon), np.nan),
        'mean_intensity_max': np.full((n_lat, n_lon), np.nan),
        'max_intensity': np.full((n_lat, n_lon), np.nan),
        'total_cumulative': np.full((n_lat, n_lon), np.nan),
        'mean_rate_onset': np.full((n_lat, n_lon), np.nan),
        'mean_rate_decline': np.full((n_lat, n_lon), np.nan)
    }
    
    # Process each valid location
    print(f"\nProcessing {len(valid_locations):,} valid locations...")
    print("Progress will be saved every 500 locations.\n")
    
    successful = 0
    failed = 0
    
    for idx, (i, j) in enumerate(tqdm(valid_locations, desc="Processing")):
        # Get time series for this location
        sst_ts, time_coords = get_timeseries_for_location(nc_files, data_dir, i, j)
        
        # Run MHW detection
        result = detect_mhw_at_location(sst_ts, time_coords, clim_period)
        
        if result is not None:
            successful += 1
            for key in results.keys():
                results[key][i, j] = result[key]
        else:
            failed += 1
        
        # Save checkpoint every 1000 locations
        if (idx + 1) % 500 == 0:
            checkpoint_file = os.path.join(data_dir, 'mhw_checkpoint.pkl')
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'processed': idx + 1,
                    'successful': successful,
                    'failed': failed,
                    'valid_locations': valid_locations,
                    'current_idx': idx
                }, f)
            print(f"\n  Checkpoint saved: {idx+1:,}/{len(valid_locations):,} locations")
            print(f"  Successful: {successful:,} | Failed: {failed:,}")
    
    print(f"\n✓ Processing complete!")
    print(f"  Valid locations: {len(valid_locations):,}")
    print(f"  Successful: {successful:,}")
    print(f"  Failed: {failed:,}")
    print(f"  Success rate: {100*successful/len(valid_locations):.1f}%")
    
    # Create xarray Dataset
    print("\nCreating xarray Dataset...")
    mhw_ds = xr.Dataset(
        {
            'n_events': (['latitude', 'longitude'], results['n_events']),
            'total_mhw_days': (['latitude', 'longitude'], results['total_days']),
            'mean_duration': (['latitude', 'longitude'], results['mean_duration']),
            'max_duration': (['latitude', 'longitude'], results['max_duration']),
            'mean_intensity_max': (['latitude', 'longitude'], results['mean_intensity_max']),
            'max_intensity': (['latitude', 'longitude'], results['max_intensity']),
            'total_cumulative_intensity': (['latitude', 'longitude'], results['total_cumulative']),
            'mean_rate_onset': (['latitude', 'longitude'], results['mean_rate_onset']),
            'mean_rate_decline': (['latitude', 'longitude'], results['mean_rate_decline'])
        },
        coords={
            'latitude': lat,
            'longitude': lon
        },
        attrs={
            'description': 'Marine heatwave metrics for Alaska EEZ (gridded, streaming)',
            'method': 'marineHeatWaves package (Hobday et al. 2016)',
            'climatology_period': f'{clim_period[0]}-{clim_period[1]}',
            'threshold_percentile': 90,
            'min_duration': 5,
            'max_gap': 2,
            'processing_method': 'streaming (one location at a time)',
            'valid_locations_processed': len(valid_locations),
            'successful_locations': successful
        }
    )
    
    return mhw_ds, lat, lon

# ============================================
# VISUALIZATION
# ============================================

def save_and_visualize(mhw_ds, data_dir='src/climate/sst_data/alaska'):
    """
    Save results and create visualizations.
    """
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save to NetCDF
    output_file = os.path.join(data_dir, 'alaska_mhw_gridded_streaming.nc')
    mhw_ds.to_netcdf(output_file)
    print(f"\n✓ Saved to: {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for var in mhw_ds.data_vars:
        data = mhw_ds[var].values
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            print(f"\n{var}:")
            print(f"  Mean:   {np.mean(valid_data):.2f}")
            print(f"  Median: {np.median(valid_data):.2f}")
            print(f"  Min:    {np.min(valid_data):.2f}")
            print(f"  Max:    {np.max(valid_data):.2f}")
            print(f"  Valid pixels: {len(valid_data):,}")
    
    # Create maps
    plot_gridded_mhw_maps(mhw_ds)

def plot_gridded_mhw_maps(mhw_ds, output_dir='src/climate/sst_data/alaska/plots'):
    """
    Create spatial maps of MHW metrics.
    """
    import geopandas as gpd
    from pyproj import Transformer
    
    print("\n" + "="*60)
    print("CREATING SPATIAL MAPS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load shapefile
    gdf = gpd.read_file("src/climate/shapefiles/Alaska_EEZ_clipped.shp")
    gdf = gdf.to_crs("EPSG:3338")
    
    # Transform coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
    lon = mhw_ds['longitude'].values
    lat = mhw_ds['latitude'].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x_grid, y_grid = transformer.transform(lon_grid, lat_grid)
    
    # Plot configurations
    plot_configs = [
        ('n_events', 'Number of MHW Events', 'YlOrRd', (0, 100)),
        ('total_mhw_days', 'Total MHW Days', 'Reds', (0, 1000)),
        ('mean_duration', 'Mean Event Duration (days)', 'plasma', (5, 30)),
        ('max_duration', 'Maximum Event Duration (days)', 'hot', (5, 100)),
        ('mean_intensity_max', 'Mean Maximum Intensity (°C)', 'RdYlBu_r', (0, 4)),
        ('max_intensity', 'Maximum Intensity (°C)', 'turbo', (0, 8)),
        ('total_cumulative_intensity', 'Total Cumulative Intensity (°C·days)', 'inferno', (0, 3000))
    ]
    
    for var_name, title, cmap, vrange in plot_configs:
        print(f"  Creating {var_name} map...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        data = mhw_ds[var_name].values
        
        pcm = ax.pcolormesh(x_grid, y_grid, data,
                           cmap=cmap,
                           vmin=vrange[0], vmax=vrange[1],
                           shading='auto')
        
        gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
        cbar.set_label(title, fontsize=12)
        
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_title(f'Alaska EEZ: {title}\n[Alaska Albers Projection]', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.ticklabel_format(style='plain', axis='both')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'mhw_gridded_{var_name}_streaming.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print("\n✓ All maps created!")
    
    # Create summary 4-panel figure
    create_summary_4panel(mhw_ds, x_grid, y_grid, gdf, output_dir)

def create_summary_4panel(mhw_ds, x_grid, y_grid, gdf, output_dir):
    """
    Create 4-panel summary figure.
    """
    print("  Creating 4-panel summary...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    configs = [
        ('n_events', 'Number of Events', 'YlOrRd', (0, 100)),
        ('total_mhw_days', 'Total MHW Days', 'Reds', (0, 1000)),
        ('mean_duration', 'Mean Duration (days)', 'plasma', (5, 30)),
        ('max_intensity', 'Max Intensity (°C)', 'turbo', (0, 8))
    ]
    
    for ax, (var_name, title, cmap, vrange) in zip(axes.flat, configs):
        data = mhw_ds[var_name].values
        
        pcm = ax.pcolormesh(x_grid, y_grid, data,
                           cmap=cmap, vmin=vrange[0], vmax=vrange[1],
                           shading='auto')
        
        gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
        
        plt.colorbar(pcm, ax=ax, shrink=0.7, label=title)
        
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.ticklabel_format(style='plain', axis='both')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Alaska EEZ Marine Heatwave Summary', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'mhw_gridded_summary_4panel_streaming.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================
# MAIN EXECUTION
# ============================================

def main(data_dir='src/climate/sst_data/alaska', test_mode=True, n_test_points=5):
    """
    Main execution function.
    
    Parameters:
        data_dir: path to data directory
        test_mode: if True, run on small sample; if False, run full analysis
        n_test_points: number of locations to test (only used if test_mode=True)
    """
    if test_mode:
        print("\n🧪 RUNNING IN TEST MODE 🧪")
        test_workflow(data_dir, n_test_points)
    else:
        print("\n🚀 RUNNING FULL ANALYSIS 🚀")
        print("\nThis will take ~30-60 minutes.")
        print("Press Ctrl+C to cancel, or Enter to continue...")
        input()
        
        mhw_ds, lat, lon = process_all_locations_streaming(data_dir)
        save_and_visualize(mhw_ds, data_dir)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nOutput files:")
        print(f"  - {data_dir}/alaska_mhw_gridded_streaming.nc")
        print(f"  - {data_dir}/plots/mhw_gridded_*_streaming.png")

# Run it!
if __name__ == "__main__":
    # Install tqdm if needed: pip install tqdm --break-system-packages
    
    # TEST MODE - Run on 5 random locations first
    #main(test_mode=True, n_test_points=5)
    
    # Once test passes, run full analysis by uncommenting:
    main(test_mode=False)