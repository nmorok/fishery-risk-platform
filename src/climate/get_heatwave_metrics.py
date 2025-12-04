'''
get marine heatwave metrics for each event
'''

from altair import Polygon
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import os
import sqlite3
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon, MultiPolygon
import pyproj
from pathlib import Path

# get data from sst_anomalies_metadata sql table. 
# format should be: shapefile | fishery_start | fishery_end | used_start
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

def get_sst_metadata_for_event(event_id):
    conn = sqlite3.connect('data/fishery_disasters.db')
    query = """
    SELECT shapefile, fishery_start, fishery_end, used_start
    FROM sst_anomalies_metadata
    WHERE disaster_id = ?;
    """
    df = pd.read_sql_query(query, conn, params=(event_id,))
    conn.close()
    return df

# use the shapefile to mask the sst data from the given fishery_start to fihsery_end

# the possible shapefiles are:

# - alaska
# - washington
# - oregon
# - california
# - bc
# - washington_bc
# - west_coast
# - tanner
# - pacific_cod
# - snow crab
# - snowcrab and king crab
# - norton sound red king crab


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
        #'bering sea red king crab': {
        #    'mask': 'src/climate/output_masks/beringsea_redking_crab_spatial_mask.nc',
        #    'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv',
        #    'csv_output_path': 'data/csv/alaska_mhw_events_bering_sea_redking_crab_filtered.csv'
        #},
        'Peconic estuary': {
            'mask': 'src/climate/output_masks/peconic_estuary_spatial_mask.nc',
            'sst_data_path': 'src/climate/sst_data/peconic_bay/peconic_bay_mhw_events.csv',
            'csv_output_path': 'data/csv/peconic_bay_mhw_events_peconic_estuary_filtered.csv'
        }
    }
# function to get the marine heatwave metrics for each event



def get_heatwave_metrics(event_id, region, EVENT_START, EVENT_END, USED_START=1, BASELINE_YEARS=2):

    MASK_FILE = shapefile_dict[region]['mask']
    CSV_INPUT_PATH = shapefile_dict[region]['sst_data_path']
    
    # Get base path and add event_id before extension
    base_path = Path(shapefile_dict[region]['csv_output_path'])
    #CSV_OUTPUT_PATH = str(base_path.with_stem(f"{base_path.stem}_event_{event_id}")) # for specific events
    CSV_OUTPUT_PATH = base_path# for the whole spatial mask and time frame. 


    LAT_COLUMN = "latitude"
    LON_COLUMN = "longitude"

    # Temporal filtering
    START_DATE_COLUMN = "start_date"  # Update to your date column name
    END_DATE_COLUMN = "end_date"  



    # ===== MAIN SCRIPT =====

    print("="*70)
    print("FILTERING CSV: SPATIAL + TEMPORAL")
    print("="*70)

    # ===== SPATIAL FILTERING =====
    print("\n1. Loading original mask file...")
    mask_ds = xr.open_dataset(MASK_FILE)
    mask = mask_ds['mask']

    print(f"   Mask shape: {mask.shape}")
    print(f"   Lat range: [{float(mask.latitude.min()):.4f}, {float(mask.latitude.max()):.4f}]")
    print(f"   Lon range: [{float(mask.longitude.min()):.4f}, {float(mask.longitude.max()):.4f}]")
    print(f"   Cells inside EEZ: {mask.notnull().sum().values}")

    print("\n2. Loading CSV...")
    df = pd.read_csv(CSV_INPUT_PATH)
    print(f"   Total rows: {len(df)}")
    print(f"   Unique coordinates: {len(df[[LAT_COLUMN, LON_COLUMN]].drop_duplicates())}")

    print("\n3. Filtering CSV using spatial mask...")

    # Create a set of valid (lat, lon) coordinates from the mask
    valid_coords = set()
    mask_values = mask.values
    lat_array = mask.latitude.values
    lon_array = mask.longitude.values

    for i, lat in enumerate(lat_array):
        for j, lon in enumerate(lon_array):
            if not np.isnan(mask_values[i, j]):
                valid_coords.add((lat, lon))

    print(f"   Valid coordinates from mask: {len(valid_coords)}")

    # Filter the dataframe spatially
    df_spatial = df[df.apply(
        lambda row: (row[LAT_COLUMN], row[LON_COLUMN]) in valid_coords,
        axis=1
    )].copy()

    print(f"\n   === SPATIAL FILTERING RESULTS ===")
    print(f"   Original rows: {len(df)}")
    print(f"   After spatial filter: {len(df_spatial)}")
    print(f"   Percentage retained: {100 * len(df_spatial) / len(df):.2f}%")

    filtered_unique = df_spatial[[LAT_COLUMN, LON_COLUMN]].drop_duplicates()
    original_unique = df[[LAT_COLUMN, LON_COLUMN]].drop_duplicates()
    print(f"   Original unique coords: {len(original_unique)}")
    print(f"   Filtered unique coords: {len(filtered_unique)}")

    if len(filtered_unique) == len(original_unique):
        print("   ✓ PERFECT MATCH! All coordinates from CSV are in the mask!")

    # ===== TEMPORAL FILTERING =====
    print("\n4. Applying temporal filter...")

    # Convert dates
    df_spatial[START_DATE_COLUMN] = pd.to_datetime(df_spatial[START_DATE_COLUMN])
    df_spatial[END_DATE_COLUMN] = pd.to_datetime(df_spatial[END_DATE_COLUMN])
    event_start = pd.to_datetime(EVENT_START)
    event_end = pd.to_datetime(EVENT_END)

    # Calculate baseline cutoff and filter
    if USED_START == 1:
        baseline_cutoff = event_start - pd.DateOffset(years=BASELINE_YEARS)
        df_filtered = df_spatial[
            (df_spatial[START_DATE_COLUMN] >= baseline_cutoff) &
            (df_spatial[END_DATE_COLUMN] <= event_start)
        ].copy()
        print(f"   Using event START date: {event_start.date()}")
    else:
        baseline_cutoff = event_end - pd.DateOffset(years=BASELINE_YEARS)
        df_filtered = df_spatial[
            (df_spatial[START_DATE_COLUMN] >= baseline_cutoff) &
            (df_spatial[END_DATE_COLUMN] <= event_end)
        ].copy()
        print(f"   Using event END date: {event_end.date()}")

    print(f"   Baseline cutoff: {baseline_cutoff.date()}")
    print(f"   Time window: {baseline_cutoff.date()} to {(event_start if USED_START == 1 else event_end).date()}")

    print(f"\n   === TEMPORAL FILTERING RESULTS ===")
    print(f"   After spatial filter: {len(df_spatial)}")
    print(f"   After temporal filter: {len(df_filtered)}")
    print(f"   Percentage retained: {100 * len(df_filtered) / len(df_spatial):.2f}%")

    # ===== FINAL SUMMARY =====
    print(f"\n5. Final Summary:")
    print(f"   Original rows: {len(df)}")
    print(f"   After spatial filter: {len(df_spatial)} ({100 * len(df_spatial) / len(df):.2f}%)")
    print(f"   After temporal filter: {len(df_filtered)} ({100 * len(df_filtered) / len(df):.2f}%)")
    print(f"   Total removed: {len(df) - len(df_filtered)} rows")

    print(f"\n6. Saving to {CSV_OUTPUT_PATH}...")
    df_filtered.to_csv(CSV_OUTPUT_PATH, index=False)
    print("   ✓ Done!")

    mask_ds.close()

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

    return df_filtered


def summary_statistics(df, mask_ds, region):
    '''
    function to calculate the summary statistics used in the model
    
    summary statistics: 
    - % in heatwave: what percentage of points within the shapefile were in heatwave at least once during the time window.
    this can be changed to incorporate other qualifiers: what % of points had a heatwave with at least severity x or duration y, etc.
    - summed cumulative intensity
    - mean cumulative intensity
    - median cumulative intensity
    - summed duration
    - mean duration
    - median duration
    - max intensity_max

    '''

     # Get total number of valid grid cells in the mask
    mask = mask_ds['mask']
    total_cells = mask.notnull().sum().values
    
    # Get unique (lat, lon) coordinates that experienced heatwaves
    heatwave_coords = df[['latitude', 'longitude']].drop_duplicates()
    cells_with_heatwave = len(heatwave_coords)
    
    # Calculate percentage
    percent_in_heatwave = (cells_with_heatwave / total_cells) * 100
    
    # Calculate other statistics
    summary_stats = {
        'region': region,
        'percent_in_heatwave': percent_in_heatwave,
        'total_cells': int(total_cells),
        'cells_with_heatwave': cells_with_heatwave,
        
        # Intensity metrics
        'sum_cumulative_intensity': df['intensity_cumulative'].sum(),
        'mean_cumulative_intensity': df['intensity_cumulative'].mean(),
        'median_cumulative_intensity': df['intensity_cumulative'].median(),
        
        # Duration metrics
        'duration_days': df['duration'].sum(),
        'mean_duration': df['duration'].mean(),
        'median_duration': df['duration'].median(),
        
        # Peak metrics
        'peak_intensity': df['intensity_max'].max(),  # or 'peak_intensity' if that's your column
        
        # Event count
        'total_events': len(df)
    }
    
    return summary_stats



# Example usage:
# not 90 right now.
if __name__ == "__main__":
    event_id = 0
    #metadata_df = get_sst_metadata_for_event(event_id)
    print("SST Metadata for Event ID", event_id)
    #print(metadata_df)

    #region = metadata_df.loc[0, 'shapefile']
    
    # Get heatwave metrics
    #df = get_heatwave_metrics(
    #    event_id=event_id,
    #    region=region,
    #    EVENT_START=metadata_df.loc[0, 'fishery_start'],
    #    EVENT_END=metadata_df.loc[0, 'fishery_end'],
    #    USED_START=metadata_df.loc[0, 'used_start']
    #)
    
    # Load mask for summary statistics
    #mask_ds = xr.open_dataset(shapefile_dict[region]['mask'])
    
    # Calculate summary statistics
    #summary = summary_statistics(df, mask_ds, region)
    
    #print("\n=== SUMMARY STATISTICS ===")
    #for key, value in summary.items():
    #    print(f"{key}: {value}")
    
    
    # Save summary to SQL
    #conn = sqlite3.connect('data/fishery_disasters.db')
    #summary_df = pd.DataFrame([summary])
    #summary_df['event_id'] = event_id
    #summary_df.to_sql('heatwave_metrics', conn, if_exists='append', index=False)
    #conn.close()
    
    #mask_ds.close()

    get_heatwave_metrics(
        event_id=1,
        region = 'Washington',
        EVENT_START='2024-12-31',
        EVENT_END='2024-12-31'
    )
    
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