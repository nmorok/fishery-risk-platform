# Predicting Congressional Fishery Disaster Appropriations from Marine Heatwave Metrics

**PhD Thesis Research** | Marine Science & Policy Analysis

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Stan](https://img.shields.io/badge/Stan-Bayesian_Modeling-red.svg)](https://mc-stan.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the complete analytical framework for my PhD dissertation examining how marine heatwave characteristics influence federal disaster appropriations for fishery disasters. The research develops a **Bayesian regression model** to predict Congressional funding decisions and uses **Monte Carlo simulation** to project future appropriation needs under various climate change scenarios.

**Research Questions:**
1. How do heatwave intensity, duration, and spatial coverage relate to disaster appropriations?
2. Can we predict future appropriation needs based on projected climate scenarios?
3. How do regional and fishery-specific factors influence funding decisions?

**Key Findings:**
- Marine heatwave characteristics explain ~XX% of variance in appropriations
- A 30% increase in heatwave frequency could increase mean appropriations by 43%
- Significant regional differences in funding sensitivity to environmental variables

---

## Key Features

### Bayesian Regression Framework
- **8,000 posterior samples** from fitted Stan model
- Predictors: peak intensity, duration, spatial coverage, fishery value
- Regional fixed effects for 5 US fishing regions
- Robust handling of zero-inflated appropriation data

### Monte Carlo Simulation Engine
- **10,000+ simulations** per scenario
- Climate scenario modeling (baseline, moderate, high impact)
- Parametric distribution fitting with extrapolation capability
- Zero-truncated Poisson for disaster frequency modeling

### Interactive Dashboard
- **4-page Streamlit application** for data exploration
- Spatial-temporal heatwave visualization with day-by-day mapping
- Historical disaster timeline analysis
- Real-time climate scenario comparison
- Distribution explorer for all model variables

### Comprehensive Spatial Analysis
- **14 regional/species combinations** analyzed
- Integration with marine heatwave detection algorithms
- Spatial masking for precise regional attribution
- Support for both EEZ boundaries and ecological regions

---

## Repository Structure

```
fishery-risk-platform/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (optional)
├── .gitignore                        # Git ignore rules
│
├── data/                             # Data files (see Data section)
│   ├── csv/                          # Processed datasets
│   │   ├── model_data.csv           # Main modeling dataset
│   │   ├── *_mhw_events*.csv        # Regional heatwave data
│   │   └── *_filtered.csv           # Spatially masked events
│   ├── netcdf/                       # SST and spatial data
│   │   ├── *_sst.nc                 # Sea surface temperature
│   │   └── *_mask.nc                # Regional masks
│   ├── shapefiles/                   # Spatial boundaries
│   └── raw/                          # Original NOAA/NMFS data
│
├── models/                           # Statistical models
│   ├── stan/                         
│   │   └── bayesian_model.stan      # Stan model specification
│   ├── fitted/
│   │   ├── posterior_samples.csv    # 8,000 MCMC samples
│   │   └── model_diagnostics.txt    # Convergence metrics
│   └── distributions/
│       └── fitted_params.csv        # Parametric fits
│
├── src/                              # Source code
│   ├── data_processing/
│   │   ├── load_data.py             # Data loading utilities
│   │   ├── spatial_masking.py       # Regional filtering
│   │   └── heatwave_metrics.py      # MHW calculation
│   ├── modeling/
│   │   ├── fit_distributions.py     # Distribution fitting
│   │   ├── run_stan_model.py        # Bayesian estimation
│   │   └── monte_carlo.py           # Simulation engine
│   ├── visualization/
│   │   └── plotting_utils.py        # Custom plot functions
│   └── frontend/
│       └── streamlit_dashboard.py   # Interactive app
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_monte_carlo_testing.ipynb
│   └── 04_results_visualization.ipynb
│
├── outputs/                          # Generated outputs
│   ├── figures/                      # Publication figures
│   ├── tables/                       # Summary statistics
│   └── simulations/                  # Scenario results
│
└── docs/                             # Documentation
    ├── data_dictionary.md            # Variable descriptions
    ├── methods.md                    # Methodological details
    └── API.md                        # Code documentation
```

---

## Getting Started

### Prerequisites

- **Python 3.9+**
- **CmdStan** (for Bayesian modeling)
- **Git LFS** (for large data files, optional)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/nmorok/fishery-risk-platform.git
cd fishery-risk-platform
```

2. **Create virtual environment:**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda env create -f environment.yml
conda activate fishery-risk
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install Stan (if needed):**
```bash
pip install cmdstanpy
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

### Quick Start

**Launch the interactive dashboard:**
```bash
streamlit run src/frontend/streamlit_dashboard.py
```

**Run Monte Carlo simulation:**
```python
from src.modeling.monte_carlo import run_monte_carlo_simulation

results = run_monte_carlo_simulation(
    fitted_distributions,
    posterior_samples,
    scaling_params,
    n_simulations=10000
)
```

**Fit Bayesian model:**
```bash
python src/modeling/run_stan_model.py --data data/csv/model_data.csv
```

---

## Data

### Data Sources

1. **Fishery Disasters (1992-2023)**
   - Source: NOAA Fisheries Disaster Declarations
   - Variables: Appropriation amounts, dates, regions, species
   - N = XX disasters

2. **Marine Heatwaves (1982-2023)**
   - Source: NOAA OISST v2.1 (0.25° resolution)
   - Detection: Hobday et al. (2016) algorithm
   - Metrics: Intensity, duration, spatial coverage

3. **Fishery Economics**
   - Source: NMFS Commercial Landings
   - Ex-vessel values by region and species
   - Annual time series (1992-2023)

### Regional Coverage

- **Alaska:** Bering Sea, Gulf of Alaska
- **West Coast:** Washington, Oregon, California
- **Northeast:** New England, Mid-Atlantic
- **Southeast:** South Atlantic, Gulf of Mexico
- **Pacific Islands:** Hawaii, American Samoa, Guam

### Data Privacy

**Note:** Raw appropriation data and some NOAA datasets are not included in this public repository due to size and licensing restrictions. Contact the author for data access requests.

**Included in repo:**
- Processed model_data.csv (aggregated, anonymized)
- Summary statistics and fitted distributions
- Example heatwave events for demonstration

**Not included (available on request):**
- Full netCDF SST files (>50GB)
- Complete heatwave event catalog
- Raw NOAA disaster documentation

---

## Methodology

### Bayesian Regression Model

**Model Specification:**
```stan
Y ~ Normal(μ, σ)
μ = α + β₁·intensity + β₂·duration + β₃·coverage + β₄·value + region_effects
```

**Priors:**
- Coefficients: Normal(0, 1) - weakly informative
- Intercept: Normal(0, 10)
- Sigma: Half-Normal(0, 5)

**Estimation:**
- Algorithm: NUTS (No-U-Turn Sampler)
- Chains: 4
- Iterations: 2,000 per chain (1,000 warmup)
- Final samples: 8,000 posterior draws

### Monte Carlo Simulation

**Process:**
1. Fit parametric distributions to predictor variables
2. Sample heatwave characteristics from fitted distributions
3. Draw regression coefficients from posterior
4. Predict appropriations for each simulation
5. Aggregate to annual totals
6. Repeat 10,000 times

**Climate Scenarios:**
- **Baseline:** Historical distributions (1992-2023)
- **Moderate (RCP 4.5):** +30% frequency, +20% intensity
- **High (RCP 8.5):** +50% frequency, +40% intensity

---

## Key Results

### Model Performance

- **R² = 0.XX** (in-sample fit)
- **RMSE = $XX million** (cross-validation)
- **All parameters:** R̂ < 1.01 (excellent convergence)
- **Effective sample size:** >4,000 for all parameters

### Predictor Importance

| Variable | Coefficient | 95% CI | Interpretation |
|----------|-------------|--------|----------------|
| Peak Intensity | β₁ = X.XX | [X.XX, X.XX] | +1°C → +$XX million |
| Duration | β₂ = X.XX | [X.XX, X.XX] | +100 days → +$XX million |
| Coverage | β₃ = X.XX | [X.XX, X.XX] | +10% area → +$XX million |
| Fishery Value | β₄ = X.XX | [X.XX, X.XX] | +$100M landings → +$XX million |

### Climate Projections

**Mean Annual Appropriations (2024-2050):**
- Baseline: $111 million (95% CI: $XX-$XX million)
- Moderate: $159 million (95% CI: $XX-$XX million) - **+43%**
- High: $XXX million (95% CI: $XX-$XX million) - **+XX%**

**Probability of Exceeding $100M/year:**
- Baseline: 27%
- Moderate: 37% (+10 percentage points)
- High: XX% (+XX percentage points)

---

## Dashboard Features

### Page 1: Spatial Explorer
- Interactive map of heatwave events by region
- Date selector with persistence across regions
- Day-by-day event visualization
- Regional statistics and event details

### Page 2: Historical Disasters
- Timeline of all disasters (1992-2023)
- Filter by year, region, appropriation amount
- Relationship between heatwaves and funding
- Summary statistics by decade/region

### Page 3: Climate Scenarios
- **Two-pane comparison:** Baseline vs. Custom
- Real-time Monte Carlo simulation
- Adjustable climate multipliers (frequency, intensity, etc.)
- Distribution plots with delta metrics
- Probability thresholds (>$100M, >$200M)

### Page 4: Distributions
- All fitted distributions visualized
- QQ plots and goodness-of-fit tests
- Interactive parameter exploration
- Export capability for publication figures

---

//## Citation

//If you use this code or methodology in your research, please cite:

//```bibtex
//@phdthesis{morok2025fishery,
//  author = {Morok, Nicholas},
//  title = {Predicting Congressional Fishery Disaster Appropriations from Marine Heatwave Metrics},
//  school = {[Your University]},
//  year = {2025},
//  type = {PhD Dissertation}
//}
//```

---

## 🤝 Contributing

This is a PhD thesis project, but suggestions and bug reports are welcome!

**To report issues:**
1. Check existing issues first
2. Provide a minimal reproducible example
3. Include system info (OS, Python version, etc.)

**For questions about the research:**
- Open a GitHub Discussion
- Email: nmorok@uw.edu

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data Licensing:**
- NOAA data: Public domain (U.S. government work)
- Processed datasets: MIT License
- Always cite original data sources

---

//## Acknowledgments

//- **Dissertation Committee:** [Names]
//- **Funding:** [Grant/Fellowship information]
//- **Data Providers:** NOAA Fisheries, NOAA OceanWatch
//- **Software:** Stan Development Team, Python scientific stack
//- **Inspiration:** [Key papers that influenced your work]

---

## Contact

**Nikolai (Cole) Morokhovich**  
MS Candidate, Quantitative Ecology and Resource Management  
University of Washington
nmorok@uw.edu  


---

## 🔄 Version History

### v1.0.0 (Current)
- Complete Bayesian model with 8,000 posterior samples
- Monte Carlo framework with climate scenarios
- Interactive Streamlit dashboard (4 pages)
- Spatial analysis for 14 regional/species combinations
- Comprehensive documentation

### Future Development
- Fix heatwave metrics

---

**Last Updated:** December 2025
**Status:** Active Development - Thesis in Progress
