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
import gc

# ============================================
# REGION CONFIGURATIONS
# ============================================

REGIONS = {
    'alaska': {
        'data_dir': 'src/climate/sst_data/alaska',
        'shapefile': 'src/climate/shapefiles/Alaska_EEZ_clipped.shp',
        'lon_range': (-180, -130),
        'lat_range': (47, 66),
        'description': 'Alaska EEZ'
    },
    'bc': {
        'data_dir': 'src/climate/sst_data/bc',
        'shapefile': 'src/climate/shapefiles/BC_EEZ_clipped.shp',
        'lon_range': (-135, -122),
        'lat_range': (48, 55),
        'description': 'British Columbia EEZ'
    },
    'west_coast': {
        'data_dir': 'src/climate/sst_data/west_coast',
        'shapefile': 'src/climate/shapefiles/WestCoast_EEZ_clipped.shp',
        'lon_range': (-130, -117),
        'lat_range': (32, 49),
        'description': 'US West Coast EEZ'
    }
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def identify_valid_locations(nc_files, data_dir, lon_range, lat_range):
    """
    Identify which locations have valid (non-NaN) data.
    Uses h5netcdf backend which is lighter on memory.
    """
    print("\n" + "="*60)
    print("IDENTIFYING VALID LOCATIONS")
    print("="*60)
    
    # Get grid info from first file
    filepath = os.path.join(data_dir, nc_files[0])
    
    try:
        ds = xr.open_dataset(filepath, engine='h5netcdf')
    except:
        ds = xr.open_dataset(filepath)
    
    sst = ds['sst'].isel(zlev=0)
    lon_mask = (ds['longitude'] >= lon_range[0]) & (ds['longitude'] <= lon_range[1])
    lat_mask = (ds['latitude'] >= lat_range[0]) & (ds['latitude'] <= lat_range[1])
    sst_subset = sst.where(lon_mask & lat_mask, drop=True)
    
    lat = sst_subset['latitude'].values
    lon = sst_subset['longitude'].values
    n_lat, n_lon = len(lat), len(lon)
    
    # Load first time step to check for NaN
    first_sst = sst_subset.isel(time=0).values
    
    # Close file immediately
    ds.close()
    del ds, sst, sst_subset
    gc.collect()
    
    print(f"\nGrid size: {n_lat} × {n_lon} = {n_lat * n_lon:,} total locations")
    print(f"Region bounds: lon {lon_range}, lat {lat_range}")
    print("Checking which locations have valid data...")
    
    valid_locations = []
    
    for i in tqdm(range(n_lat), desc="Scanning grid"):
        for j in range(n_lon):
            if not np.isnan(first_sst[i, j]):
                valid_locations.append((i, j))
    
    print(f"\n✓ Found {len(valid_locations):,} valid locations ({100*len(valid_locations)/(n_lat*n_lon):.1f}% of grid)")
    print(f"✓ Skipping {n_lat*n_lon - len(valid_locations):,} land/masked locations")
    
    return valid_locations, lat, lon

def get_timeseries_for_location(nc_files, data_dir, lat_idx, lon_idx):
    """
    Extract time series for a single location across all years.
    """
    sst_ts = []
    time_coords = []
    
    for nc_file in nc_files:
        filepath = os.path.join(data_dir, nc_file)
        
        try:
            ds = xr.open_dataset(filepath, engine='h5netcdf')
        except:
            ds = xr.open_dataset(filepath)
        
        sst_point = ds['sst'].isel(zlev=0, latitude=lat_idx, longitude=lon_idx)
        sst_values = sst_point.values
        time_values = pd.to_datetime(ds['time'].values)
        
        sst_ts.extend(sst_values)
        time_coords.extend(time_values)
        
        ds.close()
        del ds, sst_point
    
    gc.collect()
    
    return np.array(sst_ts), np.array(time_coords)

def detect_mhw_at_location(sst_ts, time_coords, clim_period):
    """
    Run marineHeatWaves detection for a single location.
    """
    valid_mask = ~np.isnan(sst_ts)
    if valid_mask.sum() < 365:
        return None
    
    sst_clean = sst_ts[valid_mask]
    time_clean = time_coords[valid_mask]
    
    try:
        t_clean = np.array([datetime.toordinal(pd.Timestamp(dt).to_pydatetime()) 
                           for dt in time_clean])
    except:
        return None
    
    try:
        mhws, clim = mhw.detect(
            t_clean, 
            sst_clean,
            climatologyPeriod=clim_period,
            pctile=90,
            windowHalfWidth=5,
            smoothPercentile=True,
            smoothPercentileWidth=31,
            minDuration=5,
            joinAcrossGaps=True,
            maxGap=2,
            coldSpells=False
        )
        
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
        
        del mhws, clim, sst_clean, time_clean, t_clean
        gc.collect()
        
        return result
        
    except Exception as e:
        return None

# ============================================
# TEST MODE
# ============================================

def test_workflow(region_name, n_test_points=5):
    """
    Test the workflow on a small number of random locations.
    """
    if region_name not in REGIONS:
        print(f"Error: Region '{region_name}' not found.")
        print(f"Available regions: {list(REGIONS.keys())}")
        return
    
    config = REGIONS[region_name]
    data_dir = config['data_dir']
    
    print("\n" + "="*60)
    print(f"TEST MODE - {config['description']}")
    print("="*60)
    
    nc_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_masked.nc')])
    print(f"\nFound {len(nc_files)} files in {data_dir}")
    
    valid_locations, lat, lon = identify_valid_locations(
        nc_files, data_dir, config['lon_range'], config['lat_range']
    )
    
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
        
        sst_ts, time_coords = get_timeseries_for_location(nc_files, data_dir, i, j)
        
        print(f"  Time series length: {len(sst_ts)} days")
        print(f"  Date range: {time_coords[0].date()} to {time_coords[-1].date()}")
        print(f"  Valid data points: {(~np.isnan(sst_ts)).sum()} ({100*(~np.isnan(sst_ts)).sum()/len(sst_ts):.1f}%)")
        print(f"  SST range: {np.nanmin(sst_ts):.2f}°C to {np.nanmax(sst_ts):.2f}°C")
        
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
            print(f"  ✗ MHW Detection failed")
        
        del sst_ts, time_coords
        gc.collect()
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print(f"\nEstimated time for full run: {len(valid_locations) * 0.2 / 60:.0f} minutes")

# ============================================
# FULL PROCESSING
# ============================================

def process_all_locations_streaming(region_name):
    """
    Process ALL valid locations one at a time.
    """
    if region_name not in REGIONS:
        print(f"Error: Region '{region_name}' not found.")
        return None
    
    config = REGIONS[region_name]
    data_dir = config['data_dir']
    
    print("\n" + "="*60)
    print(f"FULL PROCESSING - {config['description']}")
    print("="*60)
    
    nc_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_masked.nc')])
    print(f"\nFound {len(nc_files)} files")
    print(f"Years: {nc_files[0].split('_')[1]} to {nc_files[-1].split('_')[1]}")
    
    valid_locations, lat, lon = identify_valid_locations(
        nc_files, data_dir, config['lon_range'], config['lat_range']
    )
    
    n_lat, n_lon = len(lat), len(lon)
    
    # Get climatology period
    try:
        ds = xr.open_dataset(os.path.join(data_dir, nc_files[0]), engine='h5netcdf')
    except:
        ds = xr.open_dataset(os.path.join(data_dir, nc_files[0]))
    first_year = pd.to_datetime(ds['time'].values[0]).year
    ds.close()
    
    try:
        ds = xr.open_dataset(os.path.join(data_dir, nc_files[-1]), engine='h5netcdf')
    except:
        ds = xr.open_dataset(os.path.join(data_dir, nc_files[-1]))
    last_year = pd.to_datetime(ds['time'].values[-1]).year
    ds.close()
    
    clim_period = [first_year, last_year]
    print(f"Climatology period: {clim_period[0]}-{clim_period[1]}")
    
    # Initialize result arrays
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
    
    print(f"\nProcessing {len(valid_locations):,} valid locations...")
    print("Progress will be saved every 1000 locations.\n")
    
    successful = 0
    failed = 0
    
    for idx, (i, j) in enumerate(tqdm(valid_locations, desc="Processing")):
        sst_ts, time_coords = get_timeseries_for_location(nc_files, data_dir, i, j)
        result = detect_mhw_at_location(sst_ts, time_coords, clim_period)
        
        if result is not None:
            successful += 1
            for key in results.keys():
                results[key][i, j] = result[key]
        else:
            failed += 1
        
        del sst_ts, time_coords, result
        
        if (idx + 1) % 100 == 0:
            gc.collect()
        
        if (idx + 1) % 1000 == 0:
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
            print(f"\n  Checkpoint: {idx+1:,}/{len(valid_locations):,} | Success: {successful:,} | Failed: {failed:,}")
    
    print(f"\n✓ Processing complete!")
    print(f"  Valid locations: {len(valid_locations):,}")
    print(f"  Successful: {successful:,}")
    print(f"  Failed: {failed:,}")
    
    # Create xarray Dataset
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
            'description': f'Marine heatwave metrics for {config["description"]}',
            'region': region_name,
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
    
    return mhw_ds

def save_results(mhw_ds, region_name):
    """
    Save results to NetCDF.
    """
    config = REGIONS[region_name]
    data_dir = config['data_dir']
    
    output_file = os.path.join(data_dir, f'{region_name}_mhw_gridded.nc')
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

# ============================================
# MAIN EXECUTION
# ============================================

def main(region_name, test_mode=True, n_test_points=5):
    """
    Main execution function.
    
    Parameters:
        region_name: 'alaska', 'bc', or 'west_coast'
        test_mode: if True, run on small sample; if False, run full analysis
        n_test_points: number of locations to test
    """
    if region_name not in REGIONS:
        print(f"\nError: Region '{region_name}' not found.")
        print(f"Available regions: {list(REGIONS.keys())}")
        return
    
    config = REGIONS[region_name]
    
    if test_mode:
        print(f"\n🧪 TESTING: {config['description']} 🧪")
        test_workflow(region_name, n_test_points)
    else:
        print(f"\n🚀 FULL ANALYSIS: {config['description']} 🚀")
        print("\nThis may take 20-60 minutes depending on region size.")
        print("Press Ctrl+C to cancel, or Enter to continue...")
        input()
        
        mhw_ds = process_all_locations_streaming(region_name)
        
        if mhw_ds is not None:
            save_results(mhw_ds, region_name)
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE!")
            print("="*60)

# ============================================
# RUN FOR ALL REGIONS
# ============================================

def process_all_regions(test_mode=True, n_test_points=5):
    """
    Process all three regions sequentially.
    """
    for region_name in ['alaska', 'bc', 'west_coast']:
        print("\n\n" + "="*60)
        print(f"STARTING: {REGIONS[region_name]['description']}")
        print("="*60)
        
        main(region_name, test_mode=test_mode, n_test_points=n_test_points)
        
        # Small delay between regions
        import time
        time.sleep(2)
    
    print("\n\n" + "="*60)
    print("ALL REGIONS COMPLETE!")
    print("="*60)

# ============================================
# RUN IT!
# ============================================

if __name__ == "__main__":
    # TEST MODE - Test one region
    main('west_coast', test_mode=True, n_test_points=5)
    
    # TEST MODE - Test all regions
    # process_all_regions(test_mode=True, n_test_points=5)
    
    # FULL ANALYSIS - One region
    # main('west_coast', test_mode=False)
    
    # FULL ANALYSIS - All regions
    # process_all_regions(test_mode=False)