'''
get marine heatwave metrics for each event
'''

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import os

# get data from sst_anomalies_metadata sql table. 
# format should be: shapefile | start_date | end_date | used_start

def get_sst_metadata_for_event(event_id, conn):
    query = f"""
    SELECT shapefile, start_date, end_date, used_start
    FROM sst_anomalies_metadata
    WHERE event_id = {event_id};
    """
    df = pd.read_sql_query(query, conn)
    return df

# use the shapefile to mask the sst data from the given start_date to end_date

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
    'alaska': {
        'shapefile': 'src/climate/sst_data/Alaska_EEZ_clipped.shp',
        'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv'
    },
    'washington': {
        'shapefile': 'src/climate/shapefiles/Washington_EEZ.shp',
        'sst_data_path': 'src/climate/sst_data/west_coast/west_coast_mhw_events.csv'
    },
    'oregon': {
        'shapefile': 'src/climate/shapefiles/Oregon_EEZ.shp',
        'sst_data_path': 'src/climate/sst_data/west_coast/west_coast_mhw_events.csv'
    },
    'california': {
        'shapefile': 'src/climate/shapefiles/California_EEZ.shp',
        'sst_data_path': 'src/climate/sst_data/west_coast/west_coast_mhw_events.csv'
    },
    'bc': {
        'shapefile': 'src/climate/shapefiles/BC_EEZ.shp',
        'sst_data_path': 'src/climate/sst_data/bc/bc_mhw_events.csv'
    },
    'washington_bc': {
        'shapefile': 'src/climate/shapefiles/Washington_BC_EEZ.shp'
        # come back to this one.
    },
    'west_coast': {
        'shapefile': 'src/climate/shapefiles/WestCoast_EEZ.shp',
        'sst_data_path': 'src/climate/sst_data/west_coast/west_coast_mhw_events.csv'
    },
    'tanner': {
        'shapefile': 'src/climate/shapefiles/Tanner_Crab.shp',
        'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv'
    },
    'pacific_cod': {
        'shapefile': 'src/climate/shapefiles/Alaska_Pcod.shp',
        'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv'
    },
    'snow crab': {
        'shapefile': 'src/climate/shapefiles/Snow_Crab.shp',
        'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv'
    },
    'snowcrab and king crab': {
        'shapefile': 'src/climate/shapefiles/snowcrab_rockcrab.shp',
        'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv'
    },
    'norton sound red king crab': {
        'shapefile': 'src/climate/shapefiles/Norton_Sound_RedKing_Crab.shp',
        'sst_data_path': 'src/climate/sst_data/west_coast/west_coast_mhw_events.csv'
    },
    'bering sea red king crab': {
        'shapefile': 'src/climate/shapefiles/BeringSea_RedKing_Crab.shp',
        'sst_data_path': 'src/climate/sst_data/alaska/alaska_mhw_events.csv'
    },
    'peconic_bay': {
        'shapefile': 'src/climate/shapefiles/Peconic_Estuary_Program_Boundary.shp',
        'sst_data_path': 'src/climate/sst_data/peconic_bay/peconic_bay_mhw_events.csv'
    }
}
# function to get the marine heatwave metrics for each event

def get_heatwave_metrics(region, event_start, event_end):
    # get the shapefile and the sst data path from the region
    shapefile = shapefile_dict[region]['shapefile']
    sst_data_path = shapefile_dict[region]['sst_data_path']

    # load the sst data and mask spatially with the shapefile
    sst_df = pd.read_csv(sst_data_path)
    sst_df['date'] = pd.to_datetime(sst_df['date'])

    # load the shapefile using geopandas
    shapefile_gdf = gpd.read_file(shapefile)
    shapefile_gdf = shapefile_gdf.to_crs("EPSG:4326")  # ensure it's in lat/lon
    
    # if it is alaska, deal with the dateline issue
    