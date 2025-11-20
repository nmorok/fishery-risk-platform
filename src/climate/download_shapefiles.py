# download shapefiles

import matplotlib.pyplot as plt
import geopandas as gpd
import os
from shapely.geometry import box

os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Create shapefiles directory if it doesn't exist
os.makedirs("shapefiles", exist_ok=True)

# File paths
bc_eez_path = "shapefiles/BC_EEZ.shp"
alaska_eez_path = "shapefiles/Alaska_EEZ.shp"
alaska_clipped_path = "shapefiles/Alaska_EEZ_clipped.shp"
west_coast_path = "shapefiles/WestCoast_EEZ.shp"

# Download BC EEZ if not exists
if not os.path.exists(bc_eez_path):
    print("Downloading BC EEZ...")
    url = "https://services9.arcgis.com/E9nHvoLcPMrfFwgp/arcgis/rest/services/BC_eez/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson"
    gdf = gpd.read_file(url)
    gdf.to_file(bc_eez_path)
    print("BC EEZ downloaded successfully")
else:
    print("BC EEZ already exists, skipping download")

if not os.path.exists(west_coast_path):
    print("Downloading West Coast EEZ...")
    url = "https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/useezpolygon/FeatureServer/0/query?where=1%3D1&outFields=*&f=json"
    gdf = gpd.read_file(url)
    gdf.to_file(west_coast_path)
    print("West Coast EEZ downloaded successfully")

# Download Alaska EEZ if not exists
if not os.path.exists(alaska_eez_path):
    print("Downloading Alaska EEZ...")
    url = "https://services1.arcgis.com/HVreyEzQWRewq33m/arcgis/rest/services/Alaska_IPHC_RegAreas_WebMerc/FeatureServer/59/query?where=1%3D1&outFields=*&f=geojson"
    gdf = gpd.read_file(url)
    gdf.to_file(alaska_eez_path)
    print("Alaska EEZ downloaded successfully")
else:
    print("Alaska EEZ already exists, skipping download")

# Clip Alaska EEZ if clipped version doesn't exist
if not os.path.exists(alaska_clipped_path):
    print("Clipping Alaska EEZ...")
    
    # Read the shapefile
    gdf = gpd.read_file(alaska_eez_path)
    print(f"Original CRS: {gdf.crs}")
    
    # Fix any invalid geometries first
    gdf['geometry'] = gdf['geometry'].make_valid()
    print(f"Geometries validated")
    
    # Set latitude cutoff at 65.75°N
    max_latitude = 65.75
    
    # Create a clipping box that avoids the antimeridian wrap
    # Clip to western hemisphere only (Alaska proper, not the wrap-around)
    minx, miny, maxx, maxy = -180, 47, -130, max_latitude
    print(f"Clipping bounds: {minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}")
    
    clip_box = box(minx, miny, maxx, max_latitude)
    
    # Convert clip_box to a GeoDataFrame with matching CRS
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs=gdf.crs)
    
    # Clip using overlay method (more robust)
    gdf_clipped = gpd.overlay(gdf, clip_gdf, how='intersection')
    
    # Save the result
    gdf_clipped.to_file(alaska_clipped_path)
    print(f"Done! Clipped to {max_latitude}°N. Features: {len(gdf)} → {len(gdf_clipped)}")
else:
    print("Clipped Alaska EEZ already exists, skipping clipping")

# Load shapefiles
print("\nLoading shapefiles for plotting...")
bc_eez = gpd.read_file(bc_eez_path)
alaska_eez_clipped = gpd.read_file(alaska_clipped_path)
west_coast = gpd.read_file(west_coast_path)


#reproject to alaska crs
bc_eez_proj = bc_eez.to_crs("EPSG:3338")
west_coast_proj = west_coast.to_crs("EPSG:3338")
alaska_eez_clipped_proj = alaska_eez_clipped.to_crs("EPSG:3338")

# Plot both on the same map
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot BC EEZ
bc_eez_proj.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.7, label='BC EEZ')

# Plot Clipped Alaska EEZ
alaska_eez_clipped_proj.plot(ax=ax, edgecolor='black', facecolor='lightcoral', alpha=0.7, label='Alaska EEZ')

# Plot West Coast 
west_coast_proj.plot(ax=ax, edgecolor='green', facecolor='lightgreen', alpha=0.7, label='West Coast')

ax.set_title('BC, Alaska EEZ, and West Coast', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nAll files ready!")
print(f"- BC EEZ: {len(bc_eez)} features")
print(f"- Alaska EEZ (clipped): {len(alaska_eez_clipped)} features")
print(f"- West Coast EEZ: {len(west_coast)} features")