from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_regions():
    response = client.get("/regions")
    assert response.status_code == 200
    regions = response.json()
    assert "Alaska" in regions
    assert len(regions) == 8

def test_get_species_by_region():
    response = client.get("/regions/Alaska/species")
    assert response.status_code == 200
    species = response.json()
    assert "Snow Crab" in species

def test_simulate_endpoint():
    payload = {
        "region": "Alaska",
        "species": "Snow Crab",
        "climate_scenario": "baseline",
        "n_iterations": 1000  # Small number for fast test
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pml_metrics" in data
    assert data["pml_metrics"]["PML_50yr"] > 0

def test_invalid_region():
    payload = {
        "region": "Invalid Region",
        "species": "Snow Crab",
        "climate_scenario": "baseline"
    }
    response = client.post("/simulate", json=payload)
    assert response.status_code == 400