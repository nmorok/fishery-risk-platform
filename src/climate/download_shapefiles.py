# download shapefiles

import matplotlib.pyplot as plt
import geopandas as gpd
import os
from shapely.geometry import box

os.chdir('src/climate')

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
    gdf_dissolved = gdf.dissolve()
    gdf = gdf_dissolved.reset_index()
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
    gdf_dissolved = gdf_clipped.dissolve()
    gdf_dissolved = gdf_dissolved.reset_index()
    print(f"After dissolving: {len(gdf_dissolved)} features")
    
    # Save the result
    gdf_dissolved.to_file(alaska_clipped_path)
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


# Breaking down the west coast into the three states.

Washington_EEZ_path = "shapefiles/Washington_EEZ.shp"
Oregon_EEZ_path = "shapefiles/Oregon_EEZ.shp"
California_EEZ_path = "shapefiles/California_EEZ.shp"


# First, let's check the actual bounds of the West Coast EEZ
gdf = gpd.read_file(west_coast_path)
print(f"\nWest Coast EEZ bounds (original): {gdf.total_bounds}")
print(f"West Coast EEZ CRS: {gdf.crs}")

# Convert to EPSG:4326 (lat/lon) for clipping
gdf_latlon = gdf.to_crs("EPSG:4326")
print(f"West Coast EEZ bounds (lat/lon): {gdf_latlon.total_bounds}")

# Washington is the westcoast file clipped to washington state boundary: 46.3457000 and above
if not os.path.exists(Washington_EEZ_path):
    print("Creating Washington EEZ shapefile...")
    gdf = gpd.read_file(west_coast_path).to_crs("EPSG:4326")  # Convert to lat/lon
    gdf['geometry'] = gdf['geometry'].make_valid()
    
    min_latitude = 46.3457
    bounds = gdf.total_bounds  # Now in degrees
    minx, maxx = bounds[0], bounds[2]
    
    clip_box = box(minx, min_latitude, maxx, 49)
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs="EPSG:4326")
    gdf_clipped = gpd.overlay(gdf, clip_gdf, how='intersection')
    gdf_clipped.to_file(Washington_EEZ_path)
    print(f"Washington EEZ created with {len(gdf_clipped)} features")
else:
    print("Washington EEZ already exists, skipping creation")

# Oregon is the westcoast file clipped between 42°N and 46.3457°N
if not os.path.exists(Oregon_EEZ_path):
    print("Creating Oregon EEZ shapefile...")
    gdf = gpd.read_file(west_coast_path).to_crs("EPSG:4326")  # Convert to lat/lon
    gdf['geometry'] = gdf['geometry'].make_valid()
    
    min_latitude = 42.0
    max_latitude = 46.3457
    bounds = gdf.total_bounds
    minx, maxx = bounds[0], bounds[2]
    
    clip_box = box(minx, min_latitude, maxx, max_latitude)
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs="EPSG:4326")
    gdf_clipped = gpd.overlay(gdf, clip_gdf, how='intersection')
    gdf_clipped.to_file(Oregon_EEZ_path)
    print(f"Oregon EEZ created with {len(gdf_clipped)} features")
else:
    print("Oregon EEZ already exists, skipping creation")

# California is the westcoast file clipped to 42°N and below
if not os.path.exists(California_EEZ_path):
    print("Creating California EEZ shapefile...")
    gdf = gpd.read_file(west_coast_path).to_crs("EPSG:4326")  # Convert to lat/lon
    gdf['geometry'] = gdf['geometry'].make_valid()
    
    max_latitude = 42.0
    bounds = gdf.total_bounds
    minx, maxx = bounds[0], bounds[2]
    
    clip_box = box(minx, 32, maxx, max_latitude)
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs="EPSG:4326")
    gdf_clipped = gpd.overlay(gdf, clip_gdf, how='intersection')
    gdf_clipped.to_file(California_EEZ_path)
    print(f"California EEZ created with {len(gdf_clipped)} features")
else:
    print("California EEZ already exists, skipping creation")

print("\nState-level EEZ files complete!")

