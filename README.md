README


fishery-risk-platform/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── synthetic/          # Generated data
│   └── outputs/            # Simulation results
├── src/
│   ├── __init__.py
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── generate_disasters.py
│   │   ├── generate_heatwaves.py
│   │   └── validate_data.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── monte_carlo.py
│   │   ├── loss_models.py
│   │   └── pml_calculator.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models.py
│   │   └── database.py
│   └── frontend/
│       ├── __init__.py
│       └── app.py
├── tests/
│   ├── test_data_generation.py
│   ├── test_simulation.py
│   └── test_api.py
└── notebooks/
    └── exploratory_analysis.ipynb


To do:
- finish adding metadata to the notes section. actually, might just drop it. Or add the title from NOAA website. 
- Combine washington and bc so that I can get the heatwaves for those regions
- build out the model 
- build out the monte carlo simulation
- build out the stan model -- might need to switch to r for that. 
- build out the dashboard 
- clean up the metadata
- write up a report for Andre