import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import xarray as xr
from pyproj import Transformer

filepath = "src/climate/sst_data/alaska/sst_1984_masked.nc"
output_dir = "src/climate/sst_data/alaska/plots"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load netcdf file
ds = xr.open_dataset(filepath)
print("\nDataset structure:")
print(ds)

# Load shapefile (already in Alaska Albers)
gdf = gpd.read_file("src/climate/shapefiles/Alaska_EEZ_clipped.shp")
gdf = gdf.to_crs("EPSG:3338")  # Ensure it's in Alaska Albers
print(f"\nShapefile CRS: {gdf.crs}")

# Remove zlev dimension
sst = ds['sst'].isel(zlev=0)
print(f"\nSST shape after removing zlev: {sst.shape}")

# Subset to Alaska region
lon_min, lat_min, lon_max, lat_max = -180, 47, -130, 66
lon_mask = (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_max)
lat_mask = (ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_max)

# Apply masks
sst_subset = sst.where(lon_mask & lat_mask, drop=True)
print(f"\nSubset SST shape: {sst_subset.shape}")

# Get coordinates
lon = sst_subset['longitude'].values
lat = sst_subset['latitude'].values

print(f"Lon range: {lon.min():.2f} to {lon.max():.2f}")
print(f"Lat range: {lat.min():.2f} to {lat.max():.2f}")

# Create transformer from WGS84 to Alaska Albers
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)

# Transform coordinates
print("\nTransforming coordinates to Alaska Albers...")
lon_grid, lat_grid = np.meshgrid(lon, lat)
x_grid, y_grid = transformer.transform(lon_grid, lat_grid)

print(f"X range: {x_grid.min()/1000:.0f} to {x_grid.max()/1000:.0f} km")
print(f"Y range: {y_grid.min()/1000:.0f} to {y_grid.max()/1000:.0f} km")

# Get statistics
sst_day1 = sst_subset.isel(time=0).values
print(f"\nDay 1 SST statistics:")
print(f"  Min: {np.nanmin(sst_day1):.2f}°C")
print(f"  Max: {np.nanmax(sst_day1):.2f}°C")
print(f"  Mean: {np.nanmean(sst_day1):.2f}°C")
print(f"  NaN percentage: {100 * np.isnan(sst_day1).sum() / sst_day1.size:.1f}%")

# ============================================
# PLOT 1: Day 1 SST map in Alaska Albers
# ============================================
print("\nCreating Day 1 SST map...")
fig, ax = plt.subplots(figsize=(14, 10))

# Plot SST using transformed coordinates
pcm = ax.pcolormesh(x_grid, y_grid, sst_day1,
                    cmap='RdYlBu_r',
                    vmin=-2,
                    vmax=20,
                    shading='auto')

# Add shapefile (already in Alaska Albers)
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)

# Colorbar
cbar = plt.colorbar(pcm, ax=ax, label='SST (°C)', shrink=0.7)

ax.set_xlabel('Easting (m)', fontsize=12)
ax.set_ylabel('Northing (m)', fontsize=12)
ax.set_title('Alaska EEZ Sea Surface Temperature - Day 1 (1984)\n[Alaska Albers Projection]', 
             fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.ticklabel_format(style='plain', axis='both')
ax.grid(True, alpha=0.3)
plt.tight_layout()

output_file = os.path.join(output_dir, 'alaska_sst_day1_1984_albers.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

