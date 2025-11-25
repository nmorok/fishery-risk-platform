import xarray as xr

# Load results
alaska_mhw = xr.open_dataset('src/climate/sst_data/alaska/alaska_mhw_gridded_streaming.nc')
#bc_mhw = xr.open_dataset('src/climate/sst_data/bc/bc_mhw_gridded_streaming.nc')
#wc_mhw = xr.open_dataset('src/climate/sst_data/west_coast/west_coast_mhw_gridded_streaming.nc')

# Quick stats
print(alaska_mhw['n_events'].mean().values)
print(alaska_mhw)
df = alaska_mhw.to_dataframe().reset_index()
df.to_csv('mhw_metrics.csv', index=False)