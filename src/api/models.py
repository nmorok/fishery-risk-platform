from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class ClimateScenario(str, Enum):
    baseline = "baseline"
    moderate = "moderate"
    severe = "severe"

class SimulationRequest(BaseModel):
    region: str = Field(..., example="Alaska")
    species: str = Field(..., example="Snow Crab")
    climate_scenario: ClimateScenario = Field(default="baseline")
    n_iterations: int = Field(default=10000, ge=1000, le=50000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "region": "Alaska",
                "species": "Snow Crab",
                "climate_scenario": "moderate",
                "n_iterations": 10000
            }
        }

class PMLMetrics(BaseModel):
    PML_50yr: float
    PML_100yr: float
    PML_250yr: float
    VaR_95: float
    VaR_99: float
    TVaR_95: float
    expected_annual_loss: float

class SimulationResponse(BaseModel):
    request: SimulationRequest
    pml_metrics: PMLMetrics
    loss_distribution: List[float]  # For histogram
    loss_exceedance_curve: Dict[str, List[float]]  # {loss: [...], prob: [...]}
    computation_time_seconds: float

class DisasterResponse(BaseModel):
    disaster_id: int
    year: int
    month: int
    region: str
    species: str
    disaster_type: str
    loss_usd: float

class RegionResponse(BaseModel):
    region: str
    available_species: List[str]
    total_disasters: int
    total_loss_usd: float




    @app.post("/simulate", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    db: Session = Depends(get_db)
):
    """
    Run Monte Carlo simulation for specified region/species/scenario
    
    This is the main endpoint that:
    1. Loads historical disaster data from database
    2. Runs Monte Carlo simulation
    3. Calculates PML and risk metrics
    4. Returns results for visualization
    """
    import time
    start_time = time.time()
    
    # Validate region and species combination
    species_in_region = db.query(SpeciesRegion).filter(
        SpeciesRegion.region == request.region,
        SpeciesRegion.species == request.species
    ).first()
    
    if not species_in_region:
        raise HTTPException(
            status_code=400,
            detail=f"{request.species} not found in {request.region}"
        )
    
    # Load historical disasters
    disasters_df = pd.read_sql(
        db.query(Disaster).filter(
            Disaster.region == request.region
        ).statement,
        db.bind
    )
    
    # Load heatwave data
    heatwaves_df = pd.read_sql(
        db.query(Heatwave).statement,
        db.bind
    )
    
    # Initialize simulator
    simulator = MonteCarloSimulator(
        disasters_df, 
        heatwaves_df,
        loss_model=None  # You'll load fitted model
    )
    
    # Run simulation
    losses = simulator.run_simulation(
        region=request.region,
        species=request.species,
        climate_scenario=request.climate_scenario,
        n_iterations=request.n_iterations
    )
    
    # Calculate metrics
    calculator = PMLCalculator(losses)
    pml_values = calculator.calculate_pml()
    var_values = calculator.calculate_var()
    tvar = calculator.calculate_tvar()
    eal = calculator.calculate_expected_loss()
    lec = calculator.calculate_loss_exceedance_curve()
    
    computation_time = time.time() - start_time
    
    return SimulationResponse(
        request=request,
        pml_metrics=PMLMetrics(
            **pml_values,
            **var_values,
            TVaR_95=tvar,
            expected_annual_loss=eal
        ),
        loss_distribution=losses.tolist(),
        loss_exceedance_curve={
            'loss': lec['loss_amount'].tolist(),
            'probability': lec['exceedance_probability'].tolist()
        },
        computation_time_seconds=computation_time
    )

@app.get("/regions", response_model=List[str])
async def get_regions(db: Session = Depends(get_db)):
    """Get list of all available regions"""
    regions = db.query(Disaster.region).distinct().all()
    return [r[0] for r in regions]

@app.get("/regions/{region}/species", response_model=List[str])
async def get_species_by_region(region: str, db: Session = Depends(get_db)):
    """Get available species for a specific region"""
    species = db.query(SpeciesRegion.species).filter(
        SpeciesRegion.region == region
    ).all()
    
    if not species:
        raise HTTPException(
            status_code=404,
            detail=f"Region '{region}' not found"
        )
    
    return [s[0] for s in species]

@app.get("/disasters", response_model=List[DisasterResponse])
async def get_disasters(
    region: Optional[str] = None,
    species: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get historical disaster data with optional filters
    """
    query = db.query(Disaster)
    
    if region:
        query = query.filter(Disaster.region == region)
    if species:
        query = query.filter(Disaster.species == species)
    if year_start:
        query = query.filter(Disaster.year >= year_start)
    if year_end:
        query = query.filter(Disaster.year <= year_end)
    
    disasters = query.limit(limit).all()
    
    return [DisasterResponse(
        disaster_id=d.disaster_id,
        year=d.year,
        month=d.month,
        region=d.region,
        species=d.species,
        disaster_type=d.disaster_type,
        loss_usd=d.loss_usd
    ) for d in disasters]

@app.get("/statistics/{region}/{species}")
async def get_statistics(
    region: str, 
    species: str,
    db: Session = Depends(get_db)
):
    """
    Get summary statistics for a region/species combination
    """
    disasters = db.query(Disaster).filter(
        Disaster.region == region,
        Disaster.species == species
    ).all()
    
    if not disasters:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for {species} in {region}"
        )
    
    losses = [d.loss_usd for d in disasters]
    
    return {
        "region": region,
        "species": species,
        "total_disasters": len(disasters),
        "total_loss_usd": sum(losses),
        "mean_loss_usd": np.mean(losses),
        "median_loss_usd": np.median(losses),
        "max_loss_usd": max(losses),
        "year_range": [min(d.year for d in disasters), 
                       max(d.year for d in disasters)]
    }