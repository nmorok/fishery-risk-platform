'''
Marine Heatwave Detection Script
Using marineHeatWaves package (Hobday et al. 2016) (included in the repo)

This script processes SST data for specified regions to identify marine heatwave events.
It supports both a test mode (random locations) and full processing mode (all locations).

It uses a streaming/iterative approach to handle the large datasets without loading everything into memory.

For a given region, it gets the list of netCDF files, identifies valid ocean locations, extracts time series for each location,
runs the marineHeatWaves detection, and saves the results to CSV. 

What the streaming means specifically: it goes through each valid location one at a time, taking the sst and date information 
from each netCDF file (yearly files) for that location, building the full time series for that point, then running the detection function.

'''


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

# changed from having more configs, now only need the data directory. 
# ============================================

REGIONS = {
    'alaska': {
        'data_dir': 'src/climate/sst_data/alaska',
        'description': 'Alaska EEZ'
    },
    'bc': {
        'data_dir': 'src/climate/sst_data/bc',
        'description': 'British Columbia EEZ'
    },
    'west_coast': {
        'data_dir': 'src/climate/sst_data/west_coast',
        'description': 'US West Coast EEZ'
    }
}

# ============================================
# Functions
# ============================================

def identify_valid_locations(nc_files, data_dir):
    """
    Identify which locations have valid (non-NaN) data -- the ocean locations.
    Assumes data is already masked to the region of interest.

    Uses the first file only to determine the 'valid'/ocean locations.
    Returns a list of (lat_idx, lon_idx) tuples for valid locations,
    """
    print("\n" + "="*60)
    print("IDENTIFYING VALID LOCATIONS")
    print("="*60)
    
    # Open the first file to get grid info
    filepath = os.path.join(data_dir, nc_files[0])
    
    # Use context manager and explicit close
    #try:
    #    with xr.open_dataset(filepath, engine='h5netcdf') as ds:
    #        sst = ds['sst'].isel(zlev=0)
    #        
    #        lat = sst['latitude'].values
    #        lon = sst['longitude'].values
    #        first_sst = sst.isel(time=0).values
    #except:
    #    with xr.open_dataset(filepath) as ds:
    #        sst = ds['sst'].isel(zlev=0)
    #        
    #        lat = sst['latitude'].values
    #        lon = sst['longitude'].values
    #        first_sst = sst.isel(time=0).values
    with xr.open_dataset(filepath) as ds:
        sst = ds['sst'].isel(zlev=0) # isel(zlev=0) may be redudant since there is only 1 depth level.
        lat = sst['latitude'].values
        lon = sst['longitude'].values
        first_sst = sst.isel(time=0).values # first time slice only to get the locations only.
    
    gc.collect() # clean up the files out of memory.
    
    n_lat, n_lon = len(lat), len(lon)
    
    # verbose output to user for debugging and size info.
    print(f"\nGrid size: {n_lat} × {n_lon} = {n_lat * n_lon:,} total locations")
    print(f"Longitude range: {lon.min():.2f} to {lon.max():.2f}")
    print(f"Latitude range: {lat.min():.2f} to {lat.max():.2f}")
    print("Checking which locations have valid data...")
    
    valid_locations = []
    
    # scanning through the whole grid looking for ocean points (non-NaN)
    for i in tqdm(range(n_lat), desc="Scanning grid"):
        for j in range(n_lon):
            if not np.isnan(first_sst[i, j]):
                valid_locations.append((i, j))
    
    print(f"\nFound {len(valid_locations):,} valid locations ({100*len(valid_locations)/(n_lat*n_lon):.1f}% of grid)")
    print(f"Skipping {n_lat*n_lon - len(valid_locations):,} land/masked locations")
    
    return valid_locations, lat, lon

def get_timeseries_for_location(nc_files, data_dir, lat_idx, lon_idx):
    """
    Extract time series for a single location across all years.
    Closing the files after reading each one to avoid memory issues.

    """
    sst_ts = []
    time_coords = []
    
    for nc_file in nc_files:
        filepath = os.path.join(data_dir, nc_file)
        
        try:
            # Open, read, and close immediately
            #try:
            #    with xr.open_dataset(filepath, engine='h5netcdf') as ds:
            #        # Select the point - this gives us a 1D array along time dimension
            #        sst_point = ds['sst'].isel(zlev=0, latitude=lat_idx, longitude=lon_idx)
            #        sst_values = sst_point.values  # This is a 1D array (time dimension)
            #        time_values = pd.to_datetime(ds['time'].values)
            #except:
            #    with xr.open_dataset(filepath) as ds:
            #        sst_point = ds['sst'].isel(zlev=0, latitude=lat_idx, longitude=lon_idx)
            #        sst_values = sst_point.values
            #        time_values = pd.to_datetime(ds['time'].values)
            
            with xr.open_dataset(filepath) as ds:
                # Select the point - this gives us a 1D array along time dimension
                sst_point = ds['sst'].isel(zlev=0, latitude=lat_idx, longitude=lon_idx)
                sst_values = sst_point.values  # This is a 1D array (time dimension)
                time_values = pd.to_datetime(ds['time'].values)

            # Both are arrays, extend the lists
            sst_ts.extend(sst_values.tolist() if hasattr(sst_values, 'tolist') else sst_values)
            time_coords.extend(time_values.tolist() if hasattr(time_values, 'tolist') else time_values)
                
        except Exception as e:
            print(f"\n  Warning: Error reading {nc_file}: {e}")
            continue
    
    # Clean up
    gc.collect()
    
    return np.array(sst_ts), np.array(time_coords)

def detect_mhw_at_location(sst_ts, time_coords, clim_period, lat_val, lon_val):
    """
    Run marineHeatWaves detection for a single location.
    Returns a list of individual events with their properties.
    """
    # Check for sufficient data
    valid_mask = ~np.isnan(sst_ts)
    if valid_mask.sum() < 365:  # Need at least 1 year of data
        return []
    
    # marineHeatWaves can handle some NaNs, but let's ensure we have mostly complete data
    if valid_mask.sum() / len(sst_ts) < 0.5:  # Less than 50% valid data
        print(f"  Warning: Insufficient valid data at ({lat_val}, {lon_val})")
        return []
    
    # Pass the data as-is (marineHeatWaves will interpolate small gaps)
    sst_clean = sst_ts
    time_clean = time_coords


    try:
        t_clean = np.array([datetime.toordinal(pd.Timestamp(dt).to_pydatetime()) 
                           for dt in time_clean])
    except:
        print(f"  Warning: Date conversion error at ({lat_val}, {lon_val})")
        return []
    
    try:
        mhws, clim = mhw.detect(
            t_clean, 
            sst_clean,
            climatologyPeriod=clim_period,
            pctile=90)

        
        # Collect individual events
        events = []
        if mhws['n_events'] > 0:
            for i in range(mhws['n_events']):
                event = {
                    'latitude': lat_val,
                    'longitude': lon_val,
                    'start_date': datetime.fromordinal(int(mhws['time_start'][i])),
                    'end_date': datetime.fromordinal(int(mhws['time_end'][i])),
                    'duration': mhws['duration'][i],
                    'intensity_max': mhws['intensity_max'][i],
                    'intensity_mean': mhws['intensity_mean'][i],
                    'intensity_cumulative': mhws['intensity_cumulative'][i]#,
                    #'rate_onset': mhws['rate_onset'][i],
                    #'rate_decline': mhws['rate_decline'][i]
                }
                events.append(event)
        
        del mhws, clim, sst_clean, time_clean, t_clean
        gc.collect()
        
        return events
        
    except Exception as e:
        return []

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
    
    # CHANGED: Remove lon_range and lat_range parameters
    valid_locations, lat, lon = identify_valid_locations(nc_files, data_dir)
    
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
        
        clim_period = [1982, 2011]
        events = detect_mhw_at_location(sst_ts, time_coords, clim_period, lat[i], lon[j])
        
        if events:
            print(f"  ✓ MHW Detection successful!")
            print(f"  Number of events: {len(events)}")
            if len(events) > 0:
                print(f"  First event: {events[0]['start_date'].date()} to {events[0]['end_date'].date()}")
                print(f"  Duration: {events[0]['duration']:.0f} days")
                print(f"  Max intensity: {events[0]['intensity_max']:.2f}°C")
        else:
            print(f"  ✗ No events detected or detection failed")
        
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
    Returns a DataFrame of all individual MHW events.
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
    
    # CHANGED: Remove lon_range and lat_range parameters
    valid_locations, lat, lon = identify_valid_locations(nc_files, data_dir)
    
    # Get climatology period
    #try:
    #    with xr.open_dataset(os.path.join(data_dir, nc_files[0]), engine='h5netcdf') as ds:
    #        first_year = pd.to_datetime(ds['time'].values[0]).year
    #except:
    #    with xr.open_dataset(os.path.join(data_dir, nc_files[0])) as ds:
    #        first_year = pd.to_datetime(ds['time'].values[0]).year
    
    #try:
    #    with xr.open_dataset(os.path.join(data_dir, nc_files[-1]), engine='h5netcdf') as ds:
    #        last_year = pd.to_datetime(ds['time'].values[-1]).year
    #except:
    #    with xr.open_dataset(os.path.join(data_dir, nc_files[-1])) as ds:
    #        last_year = pd.to_datetime(ds['time'].values[-1]).year
    
    clim_period = [1982, 2011]
    print(f"Climatology period: {clim_period[0]}-{clim_period[1]}")
    
    # Initialize list to collect all events
    all_events = []
    
    print(f"\nProcessing {len(valid_locations):,} valid locations...")
    print("Progress will be saved every 500 locations.\n")
    
    successful = 0
    failed = 0
    
    for idx, (i, j) in enumerate(tqdm(valid_locations, desc="Processing")):
        try:
            sst_ts, time_coords = get_timeseries_for_location(nc_files, data_dir, i, j)
            events = detect_mhw_at_location(sst_ts, time_coords, clim_period, lat[i], lon[j])
            
            if events:
                all_events.extend(events)
                successful += 1
            
            del sst_ts, time_coords, events
        
        except Exception as e:
            print(f"\n  Error at location ({i},{j}): {e}")
            failed += 1
        
        # Force garbage collection every 50 locations
        if (idx + 1) % 50 == 0:
            gc.collect()
        
        if (idx + 1) % 500 == 0:
            checkpoint_file = os.path.join(data_dir, 'mhw_checkpoint.pkl')
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'all_events': all_events,
                    'processed': idx + 1,
                    'successful': successful,
                    'failed': failed,
                    'current_idx': idx
                }, f)
            print(f"\n  Checkpoint: {idx+1:,}/{len(valid_locations):,} | Events: {len(all_events):,} | Success: {successful:,}")
    
    print(f"\n✓ Processing complete!")
    print(f"  Valid locations processed: {len(valid_locations):,}")
    print(f"  Locations with events: {successful:,}")
    print(f"  Total events found: {len(all_events):,}")
    print(f"  Failed: {failed:,}")
    
    # Create DataFrame
    df = pd.DataFrame(all_events)
    
    # Add metadata as attributes (though CSV won't preserve these)
    df.attrs = {
        'description': f'Marine heatwave events for {config["description"]}',
        'region': region_name,
        'method': 'marineHeatWaves package (Hobday et al. 2016)',
        'climatology_period': f'{clim_period[0]}-{clim_period[1]}',
        'threshold_percentile': 90,
        'min_duration': 5,
        'max_gap': 2,
        'valid_locations_processed': len(valid_locations),
        'locations_with_events': successful
    }
    
    return df

def save_results(df, region_name):
    """
    Save results to CSV.
    """
    if df is None or len(df) == 0:
        print("No data to save!")
        return
    
    config = REGIONS[region_name]
    data_dir = config['data_dir']
    
    output_file = os.path.join(data_dir, f'{region_name}_mhw_events.csv')
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved to: {output_file}")
    print(f"✓ Total events: {len(df):,}")
    
    # Print statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal events: {len(df):,}")
    print(f"Unique locations: {df.groupby(['latitude', 'longitude']).ngroups:,}")
    print(f"Date range: {df['start_date'].min().date()} to {df['end_date'].max().date()}")
    
    print(f"\nDuration (days):")
    print(f"  Mean:   {df['duration'].mean():.2f}")
    print(f"  Median: {df['duration'].median():.2f}")
    print(f"  Min:    {df['duration'].min():.0f}")
    print(f"  Max:    {df['duration'].max():.0f}")
    
    print(f"\nIntensity max (°C):")
    print(f"  Mean:   {df['intensity_max'].mean():.3f}")
    print(f"  Median: {df['intensity_max'].median():.3f}")
    print(f"  Max:    {df['intensity_max'].max():.3f}")
    
    print(f"\nIntensity cumulative (°C·days):")
    print(f"  Mean:   {df['intensity_cumulative'].mean():.2f}")
    print(f"  Median: {df['intensity_cumulative'].median():.2f}")
    print(f"  Max:    {df['intensity_cumulative'].max():.2f}")

# ============================================
# MAIN EXECUTION
# ============================================

def main(region_name, test_mode=True, n_test_points=5):
    """
    Main execution function.
    """
    if region_name not in REGIONS:
        print(f"\nError: Region '{region_name}' not found.")
        print(f"Available regions: {list(REGIONS.keys())}")
        return
    
    config = REGIONS[region_name]
    
    if test_mode:
        print(f"\n TESTING: {config['description']} ")
        test_workflow(region_name, n_test_points)
    else:
        print(f"\n FULL ANALYSIS: {config['description']} ")
        print("\nThis may take 20-60 minutes depending on region size.")
        print("Press Ctrl+C to cancel, or Enter to continue...")
        input()
        
        df = process_all_locations_streaming(region_name)
        
        if df is not None and len(df) > 0:
            save_results(df, region_name)
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE!")
            print("="*60)
            
            return df

def process_all_regions(test_mode=True, n_test_points=5):
    """
    Process all three regions sequentially.
    """
    for region_name in ['alaska', 'bc', 'west_coast']:
        print("\n\n" + "="*60)
        print(f"STARTING: {REGIONS[region_name]['description']}")
        print("="*60)
        
        main(region_name, test_mode=test_mode, n_test_points=n_test_points)
        
        import time
        time.sleep(2)
        gc.collect()
    
    print("\n\n" + "="*60)
    print("ALL REGIONS COMPLETE!")
    print("="*60)

def diagnose_grid_issue(data_dir):
    """There was a problem with the grid dimensions in Alaska data.  This function
    prints out the grid dimensions and lat/lon ranges for the first 3 and last 3 files
    in the specified data directory.

    The fix was to re-download the data from the source, but this function can help
    diagnose similar issues in the future.

    The issue stemmed from having points that crossed the dateline, causing lon values
    to jump from positive to negative values.
    
    """
    nc_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_masked.nc')])
    
    print("\n" + "="*60)
    print("DIAGNOSING GRID DIMENSIONS")
    print("="*60)
    
    for nc_file in nc_files[:3] + nc_files[-3:]:  # First 3 and last 3
        filepath = os.path.join(data_dir, nc_file)
        with xr.open_dataset(filepath) as ds:
            sst = ds['sst'].isel(zlev=0)
            lat = ds['latitude'].values
            lon = ds['longitude'].values
            
            print(f"\n{nc_file}:")
            print(f"  Grid: {len(lat)} lat × {len(lon)} lon")
            print(f"  Lat range: {lat.min():.2f} to {lat.max():.2f}")
            print(f"  Lon range: {lon.min():.2f} to {lon.max():.2f}")
            print(f"  Lon crosses dateline: {lon.min() < -170 and lon.max() > 170}")

# Run this
#diagnose_grid_issue('src/climate/sst_data/alaska')    

if __name__ == "__main__":
    # Run full analysis for BC
    main('alaska', test_mode=False)
    #main('alaska', test_mode=True, n_test_points=10)