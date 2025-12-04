# Data Dictionary

## model_data.csv

Main dataset used for Bayesian regression modeling.

| Variable | Type | Units | Description | Source |
|----------|------|-------|-------------|--------|
| `disaster_id` | String | - | Unique identifier for each disaster | Generated |
| `sst_year` | Integer | Years | Year of disaster declaration | NOAA Fisheries |
| `region` | String | - | Geographic region (Alaska, West_Coast, etc.) | NOAA Fisheries |
| `species` | String | - | Primary affected species or fishery | NOAA Fisheries |
| `total_appropriation` | Float | USD | Total federal appropriation amount | NOAA Fisheries |
| `peak_intensity` | Float | °C | Maximum heatwave intensity above climatology | NOAA OISST |
| `duration_days` | Float | Days | Total duration of all associated heatwaves | NOAA OISST |
| `spatial_coverage` | Float | km² | Total area affected by heatwaves | NOAA OISST |
| `percent_in_heatwave` | Float | % | Percentage of region experiencing heatwave | Calculated |
| `fishery_value` | Float | USD | Annual ex-vessel landings value | NMFS Commercial Landings |
| `declaration_date` | Date | - | Date of disaster declaration | NOAA Fisheries |

**Missing Data:**
- `total_appropriation = 0` indicates no appropriation (not missing)
- `peak_intensity = NA` indicates no heatwave detected during disaster period
- `fishery_value = NA` indicates unavailable economic data for that year/region

**Data Range:**
- Years: 1992-2023
- N = XX observations
- Regions: 5 (Alaska, West_Coast, Northeast, Southeast, Gulf_of_Mexico)

---

## Regional Heatwave Files

Files: `*_mhw_events.csv`, `*_mhw_events_*_filtered.csv`

Marine heatwave event catalog for each region.

| Variable | Type | Units | Description |
|----------|------|-------|-------------|
| `event_id` | Integer | - | Sequential event number within dataset |
| `start_date` | Date | - | First day of heatwave (temp > 90th percentile) |
| `end_date` | Date | - | Last day of heatwave |
| `duration_days` | Integer | Days | Number of consecutive days in heatwave |
| `intensity_max` | Float | °C | Maximum temperature anomaly during event |
| `intensity_mean` | Float | °C | Mean temperature anomaly during event |
| `intensity_cumulative` | Float | °C·days | Integrated intensity over duration |
| `latitude` | Float | Degrees | Centroid latitude of event |
| `longitude` | Float | Degrees | Centroid longitude of event |
| `area_km2` | Float | km² | Spatial extent of event |

**Filtering:**
- Unfiltered files (`*_mhw_events.csv`) contain all events in parent region
- Filtered files (`*_filtered.csv`) contain only events within specific EEZ/boundary
- Filtering performed using spatial masks in `data/netcdf/*_mask.nc`

---

## NetCDF Files

### SST Data (`*_sst.nc`)

Sea surface temperature data for heatwave detection.

**Dimensions:**
- `time`: Daily observations (1982-2023)
- `latitude`: 0.25° resolution
- `longitude`: 0.25° resolution

**Variables:**
- `sst`: Sea surface temperature (°C)
- `sst_anomaly`: Deviation from climatology (°C)
- `threshold_90`: 90th percentile threshold for heatwave detection (°C)

**Coordinates:**
- Latitude: [Min, Max] for region
- Longitude: [Min, Max] for region
- CRS: WGS84 (EPSG:4326)

### Spatial Masks (`*_mask.nc`)

Binary masks defining regional boundaries.

**Dimensions:**
- `latitude`: Matches SST grid
- `longitude`: Matches SST grid

**Variables:**
- `mask`: Binary indicator (1 = inside region, 0 = outside)
  - OR: NaN-based (NaN = outside, value = inside)

**Source:**
- NOAA EEZ boundaries
- Custom ecological region definitions

---

## Posterior Samples (`posterior_samples.csv`)

MCMC samples from Bayesian regression model.

| Variable | Type | Description |
|----------|------|-------------|
| `chain` | Integer | MCMC chain number (1-4) |
| `iteration` | Integer | Iteration within chain |
| `alpha` | Float | Intercept coefficient |
| `beta_intensity` | Float | Coefficient for peak_intensity |
| `beta_duration` | Float | Coefficient for duration_days |
| `beta_coverage` | Float | Coefficient for spatial_coverage |
| `beta_value` | Float | Coefficient for fishery_value |
| `region_effect_*` | Float | Regional fixed effects (5 regions) |
| `sigma` | Float | Residual standard deviation |
| `lp__` | Float | Log posterior density |

**Dimensions:**
- Chains: 4
- Iterations per chain: 2,000 (1,000 warmup)
- Total samples: 8,000
- All parameters: R̂ < 1.01

---

## Fitted Distributions (`fitted_params.csv`)

Parametric distribution parameters for Monte Carlo simulation.

| Variable | Distribution | Parameters | Notes |
|----------|--------------|------------|-------|
| `peak_intensity` | Gamma | shape, scale | Always positive |
| `duration_days` | Log-Normal | μ, σ | Right-skewed |
| `spatial_coverage` | Gamma | shape, scale | Area in km² |
| `fishery_value` | Log-Normal | μ, σ | Economic values |
| `n_disasters` | Zero-Truncated Poisson | λ | Annual frequency |

**Fitting Method:**
- Maximum likelihood estimation
- Goodness-of-fit: Kolmogorov-Smirnov test (p > 0.05 for all)
- Safety caps applied to prevent extreme outliers

---

## Region Definitions

| Region Code | Full Name | States/Areas | Fisheries |
|-------------|-----------|--------------|-----------|
| `Alaska` | Alaska | AK | Salmon, Crab, Pollock |
| `West_Coast` | West Coast | WA, OR, CA | Salmon, Dungeness Crab |
| `Northeast` | Northeast | ME, MA, RI, CT, NY, NJ | Lobster, Groundfish |
| `Southeast` | Southeast | NC, SC, GA, FL (Atlantic) | Shrimp, Snapper |
| `Gulf_of_Mexico` | Gulf of Mexico | FL, AL, MS, LA, TX | Shrimp, Oysters |

**Sub-regions:**
- `Washington` - Washington state EEZ
- `Oregon` - Oregon state EEZ
- `California` - California state EEZ
- `Tanner_Crab` - Specific fishery area
- etc.

---

## Variable Transformations

**For modeling:**
- `total_appropriation`: Log-transformed (log(y + 1))
- `fishery_value`: Log-transformed (log(x + 1))
- All predictors: Standardized (mean = 0, SD = 1)

**For simulation:**
- Climate multipliers applied to distribution parameters
- Results back-transformed to original scale

---

## Data Quality Notes

**Known Issues:**
1. Some early disasters (pre-2000) have incomplete economic data
2. Heatwave detection limited by satellite coverage (1982+)
3. Appropriation amounts may reflect multi-year allocations
4. Regional boundaries changed slightly over time

**Data Validation:**
- All appropriations cross-checked with Congressional records
- Heatwave detection validated against published literature
- Spatial masks verified against NOAA EEZ shapefiles

---

## Contact for Data Access

Full datasets available upon request:
- Email: [your.email@university.edu]
- Include: Research purpose, institutional affiliation
- Response time: 1-2 weeks

**Restricted Data:**
- Raw NOAA disaster documentation (PII concerns)
- Full netCDF archives (>50GB, available via NOAA servers)
