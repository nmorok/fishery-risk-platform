"""
Create a spatial mask for a SINGLE shapefile
Quick and simple version for one-off mask creation
"""

import geopandas as gpd
import xarray as xr
import regionmask
from pathlib import Path
from erddapy import ERDDAP

# ============================================================================
# CONFIGURATION - UPDATE THESE
# ============================================================================

# BC_EEZ.shp xx
# California_EEZ.shp
# Oregon_EEZ.shp
# Washington_EEZ.shp
# BeringSea_RedKing_Crab.shp
# Washington_BC_EEZ.shp

SHAPEFILE_PATH = "src/climate/shapefiles/Washington_BC_EEZ.shp"  # Path to your .shp file
OUTPUT_DIR = "src/climate/output_masks"  # Where to save the mask
REGION_NAME = "washington_bc_eez"  # Name for your region

# Set to True if this region crosses the International Date Line (like Alaska)
CROSSES_DATELINE = False

# Sample date for creating the mask grid (any valid date works)
SAMPLE_DATE = "2023-01-01"

# ============================================================================
# SCRIPT
# ============================================================================

print("="*70)
print(f"CREATING MASK FOR: {REGION_NAME}")
print("="*70)

# Load shapefile
print(f"\n1. Loading shapefile: {SHAPEFILE_PATH}")
gdf = gpd.read_file(SHAPEFILE_PATH)
if gdf.crs is None:
    gdf_wgs84 = gdf.set_crs("EPSG:4326")
else:
    gdf_wgs84 = gdf.to_crs("EPSG:4326")

print(f"   ✓ Loaded {len(gdf_wgs84)} features")

# Get bounds
bounds = gdf_wgs84.total_bounds
lon_min, lat_min, lon_max, lat_max = [float(x) for x in bounds]

if CROSSES_DATELINE:
    print("   ⚠️  Region crosses dateline - using full longitude range")
    lon_min, lon_max = -180.0, 180.0

print(f"\n2. Region bounds:")
print(f"   Longitude: {lon_min:.2f}° to {lon_max:.2f}°")
print(f"   Latitude:  {lat_min:.2f}° to {lat_max:.2f}°")

# Calculate area
gdf_meters = gdf_wgs84.to_crs('EPSG:3857')
gdf_meters['area_m2'] = gdf_meters.geometry.area
total_area_km2 = gdf_meters['area_m2'].sum() / 1e6
print(f"   Total area: {total_area_km2:.2f} km²")

# Create output directory
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
mask_file = output_path / f'{REGION_NAME.lower()}_spatial_mask.nc'

print(f"\n3. Output location: {mask_file}")

if mask_file.exists():
    print("   ⚠️  Mask file already exists!")
    overwrite = input("   Overwrite? (y/n): ").lower().strip()
    if overwrite != 'y':
        print("   Exiting without overwriting.")
        exit()

# Download sample SST data to get the grid
print("\n4. Downloading sample SST data to establish grid...")
e = ERDDAP(
    server="https://coastwatch.pfeg.noaa.gov/erddap",
    protocol="griddap"
)
e.dataset_id = "ncdcOisst21Agg_LonPM180"
e.griddap_initialize()

e.constraints.update({
    "time>=": f"{SAMPLE_DATE}T00:00:00Z",
    "time<=": f"{SAMPLE_DATE}T00:00:00Z",
    "latitude>=": lat_min,
    "latitude<=": lat_max,
    "longitude>=": lon_min,
    "longitude<=": lon_max,
})
e.variables = ["sst"]

ds_sample = e.to_xarray()

# Drop zlev coordinate if it exists
if 'zlev' in ds_sample.coords:
    ds_sample = ds_sample.drop_vars('zlev')

print("   ✓ Sample data downloaded")
print(f"   Grid shape: {ds_sample.sst.shape}")

# Create mask
print("\n5. Creating mask...")
mask = regionmask.mask_geopandas(
    gdf_wgs84, 
    ds_sample.longitude, 
    ds_sample.latitude
)

print("   ✓ Mask created")
print(f"   Mask shape: {mask.shape}")
print(f"   Grid cells inside region: {mask.notnull().sum().values}")
print(f"   Grid cells total: {mask.size}")
print(f"   Coverage: {100 * mask.notnull().sum().values / mask.size:.2f}%")

# Save mask
print("\n6. Saving mask...")
mask_ds = xr.Dataset({'mask': mask})
mask_ds.to_netcdf(mask_file)
print(f"   ✓ Saved to: {mask_file}")

# Cleanup
ds_sample.close()

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nYour mask is ready at: {mask_file}")
print("\nYou can now use this mask file in your CSV filtering script.")