import geopandas as gpd
import pandas as pd

SHAPEFILE1 = 'src/climate/shapefiles/Washington_EEZ.shp'
SHAPEFILE2 = 'src/climate/shapefiles/BC_EEZ.shp'
OUTPUT_SHAPEFILE = 'src/climate/shapefiles/Washington_BC_EEZ.shp'

# ===== MERGE SHAPEFILES =====
print("Loading shapefiles...")
gdf1 = gpd.read_file(SHAPEFILE1)
gdf2 = gpd.read_file(SHAPEFILE2)

print(f"Shapefile 1: {len(gdf1)} features")
print(f"Shapefile 2: {len(gdf2)} features")

# Check if CRS match
if gdf1.crs != gdf2.crs:
    print(f"Warning: CRS mismatch. Reprojecting shapefile 2 to match shapefile 1")
    gdf2 = gdf2.to_crs(gdf1.crs)

# Option 1: Simple concatenation (keeps both as separate features)
#merged = pd.concat([gdf1, gdf2], ignore_index=True)

# Option 2: Dissolve into single geometry (combines into one feature)
merged = pd.concat([gdf1, gdf2], ignore_index=True)
merged = merged.dissolve()

print(f"Merged shapefile: {len(merged)} features")

# Save the merged shapefile
print(f"Saving to {OUTPUT_SHAPEFILE}...")
merged.to_file(OUTPUT_SHAPEFILE)
print("✓ Done!")