# map of all of the shapefiles. 
import geopandas as gpd
import matplotlib.pyplot as plt
import os


os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Alaska
alaska_gdf = gpd.read_file("src/climate/shapefiles/Alaska_EEZ_clipped.shp")
# West Coast
west_coast_gdf = gpd.read_file("src/climate/shapefiles/WestCoast_EEZ.shp")
# California 
california_gdf = gpd.read_file("src/climate/shapefiles/California_EEZ.shp")
# Oregon
oregon_gdf = gpd.read_file("src/climate/shapefiles/Oregon_EEZ.shp")
# Washington
washington_gdf = gpd.read_file("src/climate/shapefiles/Washington_EEZ.shp")
# BC
bc_gdf = gpd.read_file("src/climate/shapefiles/BC_EEZ.shp")
# Norton Sound Red King Crab
norton_sound_gdf = gpd.read_file("src/climate/shapefiles/Norton_sound_RedKing_Crab.shp")
# Alaska Cod
alaska_cod_gdf = gpd.read_file("src/climate/shapefiles/Alaska_Pcod.shp")
# Bering Sea Snow Crab
bering_sea_snow_crab_gdf = gpd.read_file("src/climate/shapefiles/Snow_Crab.shp")
# Bering Sea Tanner Crab
bering_sea_tanner_crab_gdf = gpd.read_file("src/climate/shapefiles/Tanner_Crab.shp")
# Bering Sea Red King Crab
bering_sea_red_king_crab_gdf = gpd.read_file("src/climate/shapefiles/BeringSea_RedKing_Crab.shp")

# Check which shapefiles are missing CRS
print("CRS check:")
print(f"Norton Sound CRS: {norton_sound_gdf.crs}")
print(f"Alaska Cod CRS: {alaska_cod_gdf.crs}")
print(f"Snow Crab CRS: {bering_sea_snow_crab_gdf.crs}")
print(f"Tanner Crab CRS: {bering_sea_tanner_crab_gdf.crs}")
print(f"Red King Crab CRS: {bering_sea_red_king_crab_gdf.crs}")
print(f"California CRS: {california_gdf.crs}")
print(f"Oregon CRS: {oregon_gdf.crs}")
print(f"Washington CRS: {washington_gdf.crs}")


# Set CRS for any that are missing (assuming WGS84 lat/lon)
if norton_sound_gdf.crs is None:
    norton_sound_gdf = norton_sound_gdf.set_crs("EPSG:4326")
if alaska_cod_gdf.crs is None:
    alaska_cod_gdf = alaska_cod_gdf.set_crs("EPSG:4326")
if bering_sea_snow_crab_gdf.crs is None:
    bering_sea_snow_crab_gdf = bering_sea_snow_crab_gdf.set_crs("EPSG:4326")
if bering_sea_tanner_crab_gdf.crs is None:
    bering_sea_tanner_crab_gdf = bering_sea_tanner_crab_gdf.set_crs("EPSG:4326")
if bering_sea_red_king_crab_gdf.crs is None:
    bering_sea_red_king_crab_gdf = bering_sea_red_king_crab_gdf.set_crs("EPSG:4326")
if california_gdf.crs is None:
    california_gdf = california_gdf.set_crs("EPSG:4326")
if oregon_gdf.crs is None:
    oregon_gdf = oregon_gdf.set_crs("EPSG:4326")
if washington_gdf.crs is None:
    washington_gdf = washington_gdf.set_crs("EPSG:4326")

# Now set all to Alaska Albers projection
alaska_gdf = alaska_gdf.to_crs("EPSG:3338")
west_coast_gdf = west_coast_gdf.to_crs("EPSG:3338")
bc_gdf = bc_gdf.to_crs("EPSG:3338")
norton_sound_gdf = norton_sound_gdf.to_crs("EPSG:3338")
alaska_cod_gdf = alaska_cod_gdf.to_crs("EPSG:3338")
bering_sea_snow_crab_gdf = bering_sea_snow_crab_gdf.to_crs("EPSG:3338")
bering_sea_tanner_crab_gdf = bering_sea_tanner_crab_gdf.to_crs("EPSG:3338")
bering_sea_red_king_crab_gdf = bering_sea_red_king_crab_gdf.to_crs("EPSG:3338")
california_gdf = california_gdf.to_crs("EPSG:3338")
oregon_gdf = oregon_gdf.to_crs("EPSG:3338")
washington_gdf = washington_gdf.to_crs("EPSG:3338")

# Plot all shapefiles on one big plot
fig, ax = plt.subplots(figsize=(12, 10))
bc_gdf.plot(ax=ax, edgecolor='black', facecolor='lightblue', alpha=0.5, label='BC EEZ')
alaska_gdf.plot(ax=ax, edgecolor='black', facecolor='lightcoral', alpha=0.5, label='Alaska EEZ')
#west_coast_gdf.plot(ax=ax, edgecolor='black', facecolor='lightgreen', alpha=0.5, label='West Coast EEZ')
norton_sound_gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2, label='Norton Sound Red King Crab')
alaska_cod_gdf.plot(ax=ax, edgecolor='orange', facecolor='none', linewidth=2, label='Alaska Pacific Cod')
bering_sea_snow_crab_gdf.plot(ax=ax, edgecolor='purple', facecolor='none', linewidth=2, label='Bering Sea Snow Crab')
bering_sea_tanner_crab_gdf.plot(ax=ax, edgecolor='brown', facecolor='none', linewidth=2, label='Bering Sea Tanner Crab')
bering_sea_red_king_crab_gdf.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2, label='Bering Sea Red King Crab')  
california_gdf.plot(ax=ax, edgecolor='black', facecolor='lightgreen', alpha=0.5, label='California EEZ')
oregon_gdf.plot(ax=ax, edgecolor='black', facecolor='tan', alpha=0.5, label='Oregon EEZ')
washington_gdf.plot(ax=ax, edgecolor='black', facecolor='lightcyan', alpha=0.5, label='Washington EEZ')
ax.set_title('Maritime Boundaries and Fishery Regions', fontsize=16, fontweight='bold')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig('Pacific_shapefiles_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("All shapefiles plotted successfully.")
