import xarray as xr
from erddapy import ERDDAP
import geopandas as gpd
import regionmask
import os
import time
from pathlib import Path

os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# ============================================================================
# SETUP
# ============================================================================

print(os.getcwd())

print("="*70)
print("MULTI-REGION SST DOWNLOAD")
print("="*70)

# Define regions and their shapefile paths
regions = {
    'bc': {
        'name': 'BC EEZ',
        'shapefile': 'src/climate/shapefiles/BC_EEZ.shp',
        'output_dir': 'src/climate/sst_data/bc'
    },
    'alaska': {
        'name': 'Alaska EEZ',
        'shapefile': 'src/climate/shapefiles/Alaska_EEZ_clipped.shp',
        'output_dir': 'src/climate/sst_data/alaska'
    },
    'west_coast': {
        'name': 'West Coast EEZ',
        'shapefile': 'src/climate/shapefiles/WestCoast_EEZ.shp',
        'output_dir': 'src/climate/sst_data/west_coast'
    }
}

def get_region_bounds_with_dateline(gdf_wgs84, region_key):
    """
    Get proper bounds for regions that may cross the dateline.
    """
    bounds = gdf_wgs84.total_bounds
    lon_min, lat_min, lon_max, lat_max = [float(x) for x in bounds]
    
    # For Alaska, use the full global longitude range to capture dateline crossing
    if region_key == 'alaska':
        print("  ⚠️  Alaska crosses dateline - using full longitude range")
        return {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': -180.0,  # Full range
            'lon_max': 180.0,   # Full range
            'crosses_dateline': False  # Download as single request
        }
    else:
        return {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'crosses_dateline': False
        }
    
# Year range to download
START_YEAR = 1982
END_YEAR = 2024

# ============================================================================
# PROCESS EACH REGION
# ============================================================================

for region_key, region_info in regions.items():
    print("\n" + "="*70)
    print(f"PROCESSING: {region_info['name'].upper()}")
    print("="*70)
    
    # Read shapefile
    print(f"\nLoading shapefile: {region_info['shapefile']}")
    gdf = gpd.read_file(region_info['shapefile'])
    
    # Convert to WGS84 for SST data
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    print(f"✓ Loaded {len(gdf_wgs84)} features")
    
    # ============================================================
    # CHANGE #1: Use the function instead of direct bounds
    # ============================================================
    # OLD CODE (delete these lines):
    # bounds = gdf_wgs84.total_bounds
    # lon_min, lat_min, lon_max, lat_max = [float(x) for x in bounds]
    
    # NEW CODE:
    region_bounds = get_region_bounds_with_dateline(gdf_wgs84, region_key)
    
    lon_min = region_bounds['lon_min']
    lon_max = region_bounds['lon_max']
    lat_min = region_bounds['lat_min']
    lat_max = region_bounds['lat_max']
    
    print(f"Bounds:")
    print(f"  Longitude: {region_bounds['lon_min']:.2f}° to {region_bounds['lon_max']:.2f}°")
    print(f"  Latitude:  {region_bounds['lat_min']:.2f}° to {region_bounds['lat_max']:.2f}°")
    
    # Calculate area
    gdf_meters = gdf_wgs84.to_crs('EPSG:3857')
    gdf_meters['area_m2'] = gdf_meters.geometry.area
    total_area_km2 = gdf_meters['area_m2'].sum() / 1e6
    print(f"  Total area: {total_area_km2:.2f} km²")
    
    # Create output directory
    output_dir = Path(region_info['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # ========================================================================
    # CREATE MASK
    # ========================================================================
    
    print("\n" + "-"*70)
    print("CREATING SPATIAL MASK")
    print("-"*70)
    
    mask_file = output_dir / 'spatial_mask.nc'
    
    if mask_file.exists():
        print("Loading existing mask...")
        mask_ds = xr.open_dataset(mask_file)
        mask = mask_ds['mask']
        print("✓ Mask loaded")
    else:
        print("Creating new mask...")
        
        # Get a sample dataset to create mask
        e = ERDDAP(
            server="https://coastwatch.pfeg.noaa.gov/erddap",
            protocol="griddap"
        )
        e.dataset_id = "ncdcOisst21Agg_LonPM180"
        e.griddap_initialize()
        
        # Get just one day to create the mask
        e.constraints.update({
            "time>=": "2023-01-01T00:00:00Z",
            "time<=": "2023-01-01T00:00:00Z",
            "latitude>=": lat_min,
            "latitude<=": lat_max,
            "longitude>=": lon_min,
            "longitude<=": lon_max,
        })
        e.variables = ["sst"]
        
        print("  Downloading sample data for mask creation...")
        ds_sample = e.to_xarray()
        
        # Drop zlev coordinate if it exists
        if 'zlev' in ds_sample.coords:
            ds_sample = ds_sample.drop_vars('zlev')
        
        # Create mask
        print("  Creating mask...")
        mask = regionmask.mask_geopandas(
            gdf_wgs84, 
            ds_sample.longitude, 
            ds_sample.latitude
        )
        
        # Save mask for reuse
        mask_ds = xr.Dataset({'mask': mask})
        mask_ds.to_netcdf(mask_file)
        
        print(f"✓ Mask created and saved")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Grid cells inside EEZ: {mask.notnull().sum().values}")
        
        ds_sample.close()
    
    
    # ========================================================================
    # DOWNLOAD DATA YEAR-BY-YEAR
    # ========================================================================
    
    print("\n" + "-"*70)
    print(f"DOWNLOADING SST DATA ({START_YEAR}-{END_YEAR})")
    print("-"*70)
    
    # Check what years we already have
    existing_years = []
    for year_file in output_dir.glob('sst_*_masked.nc'):
        year = int(year_file.stem.split('_')[1])
        existing_years.append(year)
    existing_years.sort()
    
    if existing_years:
        print(f"Found existing data for years: {existing_years}")
    
    # Download missing years
    years_to_download = [y for y in range(START_YEAR, END_YEAR + 1) if y not in existing_years]
    
    if years_to_download:
        print(f"\nNeed to download: {len(years_to_download)} years")
        
        for year in years_to_download:
            output_file = output_dir / f"sst_{year}_masked.nc"
            
            print(f"\n  Downloading {year}...")
            start_time = time.time()
            
            try:
                # Create new ERDDAP connection for each year
                e = ERDDAP(
                    server="https://coastwatch.pfeg.noaa.gov/erddap",
                    protocol="griddap"
                )
                e.dataset_id = "ncdcOisst21Agg_LonPM180"
                
                # Initialize
                e.griddap_initialize()
                
                # Set constraints for this year
                e.constraints.update({
                    "time>=": f"{year}-01-01T00:00:00Z",
                    "time<=": f"{year}-12-31T00:00:00Z",
                    "latitude>=": lat_min,
                    "latitude<=": lat_max,
                    "longitude>=": lon_min,
                    "longitude<=": lon_max,
                })
                
                # Only get SST
                e.variables = ["sst"]
                
                # Download
                print(f"    Downloading from ERDDAP...")
                ds_year = e.to_xarray()
                
                # Drop zlev coordinate if it exists
                if 'zlev' in ds_year.coords:
                    ds_year = ds_year.drop_vars('zlev')
                
                print(f"    Downloaded {len(ds_year.time)} days")
                
                # Apply mask
                print(f"    Applying mask...")
                ds_year_masked = ds_year.where(mask.notnull())
                
                valid_points = ds_year_masked.sst.notnull().sum().values
                print(f"    Valid data points: {valid_points}")
                
                if valid_points == 0:
                    print(f"    ✗ WARNING: No valid data after masking!")
                    continue
                
                # Save with compression
                print(f"    Saving...")
                encoding = {'sst': {'zlib': True, 'complevel': 4}}
                ds_year_masked.to_netcdf(output_file, encoding=encoding)
                
                elapsed = time.time() - start_time
                file_size = output_file.stat().st_size / 1e6
                print(f"    ✓ {year} complete in {elapsed:.1f}s ({file_size:.1f} MB)")
                
                # Clean up
                ds_year.close()
                ds_year_masked.close()
                time.sleep(5)  # brief pause to be polite to server

                
            except Exception as e:
                print(f"    ✗ Error downloading {year}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n✓ {region_info['name']} download complete!")
    else:
        print(f"✓ All years already downloaded for {region_info['name']}!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("DOWNLOAD COMPLETE - SUMMARY")
print("="*70)

for region_key, region_info in regions.items():
    output_dir = Path(region_info['output_dir'])
    year_files = sorted(output_dir.glob('sst_*_masked.nc'))
    
    if year_files:
        years = [int(f.stem.split('_')[1]) for f in year_files]
        total_size = sum(f.stat().st_size for f in year_files) / 1e6
        
        print(f"\n{region_info['name']}:")
        print(f"  Directory: {output_dir}")
        print(f"  Years: {min(years)}-{max(years)} ({len(years)} files)")
        print(f"  Total size: {total_size:.1f} MB")
    else:
        print(f"\n{region_info['name']}: No data files found")

print("\n" + "="*70)