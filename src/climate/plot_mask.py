"""
Plot and visualize spatial masks created from shapefiles
"""

import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your mask file
MASK_FILE = "src/climate/output_masks/washington_bc_eez_spatial_mask.nc"

# Path to original shapefile (optional, for overlay)
SHAPEFILE_PATH = "src/climate/shapefiles/Washington_BC_EEZ.shp"

# Output filename for the plot (set to None to just display)
OUTPUT_PLOT = "src/climate/output_masks/washington_bc_eez_mask_plot.png"

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*70)
print("PLOTTING SPATIAL MASK")
print("="*70)

print("\n1. Loading mask file...")
mask_ds = xr.open_dataset(MASK_FILE)
mask = mask_ds['mask']

print(f"   ✓ Mask loaded")
print(f"   Mask shape: {mask.shape}")
print(f"   Latitude range: {float(mask.latitude.min()):.2f}° to {float(mask.latitude.max()):.2f}°")
print(f"   Longitude range: {float(mask.longitude.min()):.2f}° to {float(mask.longitude.max()):.2f}°")
print(f"   Grid cells inside region: {mask.notnull().sum().values}")
print(f"   Grid cells total: {mask.size}")
print(f"   Coverage: {100 * mask.notnull().sum().values / mask.size:.2f}%")

# Load shapefile if provided
gdf = None
if SHAPEFILE_PATH:
    print("\n2. Loading shapefile for overlay...")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    print(f"   ✓ Shapefile loaded: {len(gdf)} features")

# ============================================================================
# CREATE PLOT
# ============================================================================

print("\n3. Creating plot...")

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================================
# PLOT 1: Mask only
# ============================================================================
ax1 = axes[0]

# Plot the mask
mask_plot = mask.plot(
    ax=ax1,
    cmap='Blues',
    add_colorbar=True,
    cbar_kwargs={'label': 'Region ID', 'shrink': 0.8}
)

# Overlay shapefile boundary if available
if gdf is not None:
    gdf.boundary.plot(ax=ax1, color='red', linewidth=2, label='Shapefile boundary')
    ax1.legend(loc='upper right')

ax1.set_xlabel('Longitude (°)')
ax1.set_ylabel('Latitude (°)')
ax1.set_title('Spatial Mask - Full View')
ax1.grid(True, alpha=0.3)

# ============================================================================
# PLOT 2: Mask with data overlay
# ============================================================================
ax2 = axes[1]

# Create binary mask for better visualization
binary_mask = (~mask.isnull()).astype(float)

# Plot binary mask
im = ax2.pcolormesh(
    mask.longitude,
    mask.latitude,
    binary_mask,
    cmap='RdYlGn',
    alpha=0.6,
    shading='auto'
)

# Overlay shapefile boundary
if gdf is not None:
    gdf.boundary.plot(ax=ax2, color='darkblue', linewidth=2, label='Shapefile boundary')
    ax2.legend(loc='upper right')

ax2.set_xlabel('Longitude (°)')
ax2.set_ylabel('Latitude (°)')
ax2.set_title('Mask Coverage (Green = Inside Region)')
ax2.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label('Inside Region')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['No', 'Yes'])

plt.tight_layout()

# ============================================================================
# SAVE OR DISPLAY
# ============================================================================

if OUTPUT_PLOT:
    output_path = Path(OUTPUT_PLOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"\n4. Plot saved to: {OUTPUT_PLOT}")
else:
    print("\n4. Displaying plot...")
    plt.show()