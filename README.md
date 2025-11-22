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


    PhaseDurationKey DeliverablesResume Impact
    0: Setup 2 days Project structure, environment Foundation
    1: Data5 daysSynthetic database, validationStatistical simulation
    2: Simulation7 daysMonte Carlo engine, PML calculatorCore modeling skills
    3: API7 daysFastAPI backend, 8+ endpointsBackend development
    4: Frontend7 daysStreamlit dashboard, visualizationsFull-stack capability
    5: Deploy2 daysDocker, cloud deploymentProduction skills