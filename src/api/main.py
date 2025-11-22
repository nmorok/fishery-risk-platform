from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import sys
sys.path.append('..')

from .database import get_db, Disaster, Heatwave, SpeciesRegion
from .models import (
    SimulationRequest, SimulationResponse, 
    DisasterResponse, RegionResponse
)
from simulation.monte_carlo import MonteCarloSimulator
from simulation.pml_calculator import PMLCalculator

app = FastAPI(
    title="Fishery Disaster Risk API",
    description="API for calculating Probable Maximum Loss for U.S. fisheries",
    version="1.0.0"
)

# Add CORS middleware (so frontend can access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Fishery Disaster Risk API",
        "version": "1.0.0"
    }

from functools import lru_cache
from fastapi import BackgroundTasks

# Cache database queries
@lru_cache(maxsize=32)
def get_region_species():
    """Cache region-species combinations"""
    pass

# Add background tasks for long simulations
@app.post("/simulate/async")
async def run_simulation_async(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    For very long simulations (n_iterations > 20K),
    run in background and return job_id
    """
    job_id = str(uuid.uuid4())
    background_tasks.add_task(
        run_simulation_task, 
        job_id, 
        request, 
        db
    )
    return {"job_id": job_id, "status": "processing"}