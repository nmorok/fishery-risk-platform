# Onboarding Guide: Fishery Risk Platform

This document explains what this repo actually does and how the pieces fit together, based on reading the code itself (not the aspirational `README.md`, which describes a cleaner structure than what exists today). Use this as the map; use `README.md` for the research framing (question, findings, thesis context).

## The one-sentence version

The project asks: **do the characteristics of a marine heatwave (how hot, how long, how much area) predict how much money Congress appropriates for the resulting fishery disaster — and if heatwaves get worse under climate change, how much more money will future disasters cost?**

It answers that in three steps: build a table of historical disasters + matching heatwave stats -> fit a Bayesian regression to that table -> Monte Carlo-simulate future years under different climate scenarios using the fitted model.

## Important caveats before you touch anything

- **There is no CLI.** Every script is run with `python <script>.py` from a specific working directory (usually repo root, sometimes `src/climate/`). Several scripts have a config block of hardcoded variables near the top that you hand-edit before rerunning (e.g. which region/mask to process).
- **`src/api/models.py` is not a data-models file** — it's the actual Stan/Bayesian model-fitting script, despite the name. This is the single most confusing filename in the repo.
- **`src/api/main.py` and `tests/test_api.py` are unfinished/broken scaffolding** for a FastAPI web service that was never completed. Imports in `main.py` don't resolve (`from .database import get_db, Disaster, ...` — none of those exist in `database.py`). Don't try to run the API; the real "product" is the sequence of standalone scripts + the Streamlit dashboard.
- **`src/data_generation/`** is an abandoned early prototype (synthetic disaster/heatwave generators, most functions are empty stubs) that predates the real NOAA-data pipeline. It's unrelated to the current results — safe to ignore or delete.
- **`get_heatwave_metrics.py`** has a stray duplicated/broken block after its `__main__` section — only safe to `import` the function from it, not run the file directly. `filter_heatwaves.py` has the same content and is the one actually used.
- The README's "Repository Structure" section (directories like `models/`, `outputs/`, notebooks numbered 01-04) describes a **planned** layout that doesn't match what's on disk. Don't trust it for navigation.

## The pipeline, stage by stage

### Stage 1 — Climate data acquisition & marine heatwave detection (`src/climate/`)

Goal: turn raw satellite sea-surface-temperature data into a catalog of discrete marine heatwave events, filtered to the exact region and time window of each fishery disaster.

```
download_shapefiles.py   → builds EEZ boundary polygons (Alaska, BC, WA/OR/CA) in src/climate/shapefiles/
merge_BC_Washington.py    → combines WA + BC shapefiles where a disaster spans both
get_masks.py              → turns a shapefile into a gridded spatial mask (.nc) for one region/species at a time
                             (hand-edit SHAPEFILE_PATH / OUTPUT_DIR / REGION_NAME per run)
download_sst.py           → downloads daily NOAA OISST sea-surface-temperature data (1982–2024),
                             masked to each region → src/climate/sst_data/<region>/sst_<year>_masked.nc
heatwaves_script2.py      → runs the Hobday et al. (2016) detection algorithm (vendored in marineHeatWaves.py)
                             pixel-by-pixel over the SST time series → <region>_mhw_events.csv (one row per
                             detected heatwave per grid cell)
filter_heatwaves.py       → get_heatwave_metrics(): spatially filters events to a region's mask and temporally
                             filters to a specific disaster's date window; summary_statistics(): aggregates
                             into per-event stats (peak intensity, duration, % area affected, etc.)
                             → data/csv/<region>_mhw_events_<mask>_filtered.csv
```

`filter_heatwaves.py` also has `get_sst_metadata_for_event(event_id)`, which reads the `sst_anomalies_metadata` table in `data/fishery_disasters.db` to know which shapefile and date range to use for a given NOAA disaster — this is the bridge between the SQL database (Stage 2) and the climate scripts.

`heatwaves_script.py` (no "2") and `heatwave.py`/`plot_mask.py` are earlier/diagnostic versions — not part of the live chain.

### Stage 2 — Database & feature table (`src/api/database.py`, `create_tables.sql`)

Goal: join disaster records with their matching heatwave stats into one flat table for modeling.

- `data/fishery_disasters.db` (SQLite) holds the real data: `disasters`, `fishery_events` (appropriation $, region, species, year), `sst_anomalies_metadata` (which shapefile/mask to use for each disaster), `heatwave_metrics` (the output of Stage 1's `summary_statistics()`), plus link tables.
- `database.py` functions, run from repo root:
  - `create_tables()` — builds the schema from `create_tables.sql`
  - `load_data()` — loads each table from `data/csv/<table>.csv`
  - `create_csv()` — the key step: joins `fishery_events` + `heatwave_metrics`, computes derived columns (log-transforms, percentages, categorical fields) and writes **`data/csv/model_data.csv`** — the single feature table every downstream script reads. 38 disasters, 1992–2023.

### Stage 3 — Statistical modeling (`src/api/`)

Run in this order (each reads `model_data.csv`, applies the same dropna/filter logic):

1. **`fit_distributions.py`** — fits a parametric distribution (gamma/lognormal/normal/beta) to each predictor (peak intensity, duration, spatial coverage, fishery value) → `data/output/fitted_distributions.json`
2. **`fit_disaster_frequency.py`** — fits zero-truncated Poisson/Negative Binomial to disasters-per-year, appends to the same JSON. *(Must run after step 1 — it opens the JSON expecting it to exist.)*
3. **`models.py`** *(the actual Bayesian model, despite the filename)* — fits `model.stan` (Bayesian linear regression: `log(appropriation) ~ fishery_value + peak_intensity + duration + %area`) via `cmdstanpy` → `data/output/posterior_samples.csv`, `bayesian_model_summary.csv`, `mcmc_diagnostics.txt`
4. **`distribution_analysis.py`** — descriptive stats, correlation matrix, results-section text summary
5. **`model_analysis.py`** — post-processes the Stan output into interpretable coefficient tables, forest plots, convergence checks

### Stage 4 — Monte Carlo simulation & climate scenarios (`src/api/`)

Goal: simulate thousands of possible future years of disasters/appropriations, under different climate-change assumptions.

- **`monte_carlo_simulations.py`** — simpler/earlier version: bootstraps historical disasters directly, uses only the posterior *mean* coefficients (no uncertainty propagation). Superseded by the next two.
- **`stan_posterior_utils.py`** — more rigorous: samples predictor values from the *fitted distributions* (not bootstrap) and a random posterior draw per simulated disaster (full Bayesian uncertainty). Runs one hardcoded climate scenario.
- **`run_climate_scenarios.py`** — the final version: defines **Baseline / Moderate (+20%) / High (+40%)** climate scenarios (frequency, intensity, duration, coverage all scaled up), samples disaster frequency from the fitted Poisson/NegBinom with the scenario's multiplier, and simulates 10,000 years per scenario → `data/output/climate_scenario_{baseline,moderate,high}.csv` and `climate_scenario_comparison.csv`. **This is the one to use/extend.**

### Stage 5 — Results analysis & figures (`src/api/`)

- `simulation_analysis.py` — turns the scenario CSVs into exceedance probabilities ($50M/$100M/$200M/$500M thresholds), a baseline-vs-scenario change table, and a text report for the thesis results section.
- `visualize_distributions.py` — plots the Monte Carlo *inputs* (fitted predictor distributions vs. historical data).
- `visualize_scenarios.py` — plots how each predictor's distribution shifts across the three scenarios.
- `plot_mhw_annual_metrics.py` — independent diagnostic: heatwave metrics by year, not part of the modeling chain.

### Stage 6 — Dashboard (`src/frontend/streamlit_dashboard.py`)

A ~1600-line Streamlit app that reads only the *precomputed* outputs from Stages 1–5 (it does not recompute the model live). Four pages, picked via a sidebar radio button:

1. **Spatial Explorer** — map of a region's MHW events on a chosen date
2. **Historical Disasters** — filterable table/timeline of the 38 real disasters
3. **Climate Scenarios** — baseline results plus a UI to build a custom scenario
4. **Distributions** — predictor distributions, appropriation predictions, summary stats

`src/frontend/Hex_app/Cell_1.py` is an abandoned one-cell experiment in Hex.tech — ignore it.

## How to actually run the whole thing end-to-end

```bash
# Stage 1 (long-running, network-bound; only needed if adding new regions/species)
python src/climate/download_shapefiles.py
python src/climate/merge_BC_Washington.py
python src/climate/get_masks.py            # edit config block per mask needed
python src/climate/download_sst.py
python src/climate/heatwaves_script2.py    # edit REGIONS/__main__ per region

# Stage 2 (rebuild DB + feature table)
python -c "from src.api.database import create_tables, load_data, create_csv; create_tables(); load_data(); create_csv()"

# Stage 3 (statistical modeling, run from repo root, in order)
python src/api/fit_distributions.py
python src/api/fit_disaster_frequency.py
python src/api/models.py                   # needs CmdStan: pip install cmdstanpy && python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
python src/api/distribution_analysis.py
python src/api/model_analysis.py

# Stage 4 (simulation)
python src/api/run_climate_scenarios.py

# Stage 5 (figures/reports)
python src/api/simulation_analysis.py
python src/api/visualize_distributions.py
python src/api/visualize_scenarios.py

# Stage 6 (dashboard)
streamlit run src/frontend/streamlit_dashboard.py
```

In practice, everything through Stage 5 has already been run — `data/csv/model_data.csv` and `data/output/*.csv` are populated. A new collaborator only needs to rerun a stage if they change something upstream of it (e.g. add a new region → rerun Stage 1 onward; tweak model priors → rerun from Stage 3 onward).

## What's safe to ignore / clean up later

- `src/data_generation/` — abandoned synthetic-data prototype
- `src/api/main.py`, `tests/test_api.py` — unfinished FastAPI scaffolding
- `tests/test_data_generation.py`, `tests/test_simulation.py` — empty files
- `src/climate/heatwave.py`, `heatwaves_script.py`, `plot_mask.py`, `test.py` — one-off diagnostics, not part of the pipeline
- `src/frontend/Hex_app/` — abandoned experiment
- `notebooks/exploratory_analysis.ipynb` — early scratch work, superseded by the real scripts
- `playing_Around.R` (repo root) — scratch R script, unrelated to the Python pipeline
