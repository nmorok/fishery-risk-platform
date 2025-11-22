class MonteCarloSimulator:
    """
    Monte Carlo simulation for fishery disaster risk
    """
    
    def __init__(self, disasters_df, heatwaves_df, loss_model, seed=42):
        self.disasters = disasters_df
        self.heatwaves = heatwaves_df
        self.loss_model = loss_model
        self.seed = seed
        
    def simulate_year(self, region, species, climate_scenario='baseline'):
        """
        Simulate disasters for one year
        
        Args:
            region: Geographic region
            species: Target species
            climate_scenario: 'baseline', 'moderate', 'severe'
            
        Returns:
            Total annual loss (USD)
        """
        np.random.seed(self.seed)
        
        # Step 1: Sample number of disasters
        n_disasters = self._sample_disaster_frequency(
            region, climate_scenario
        )
        
        # Step 2: For each disaster, calculate loss
        annual_loss = 0
        for _ in range(n_disasters):
            # Sample disaster type
            disaster_type = self._sample_disaster_type(region)
            
            # Sample severity
            if disaster_type == 'Warm ocean conditions':
                severity = self._sample_heatwave_severity(
                    region, climate_scenario
                )
            else:
                severity = self._sample_other_severity(disaster_type)
            
            # Sample duration
            duration = self._sample_duration(severity)
            
            # Calculate loss
            loss = self.loss_model.predict({
                'region': region,
                'species': species,
                'severity': severity,
                'duration': duration
            })
            
            annual_loss += loss
            
        return annual_loss
        
    def run_simulation(self, region, species, climate_scenario, 
                       n_iterations=10000):
        """
        Run Monte Carlo simulation with multiple iterations
        
        Returns:
            Array of annual losses (length = n_iterations)
        """
        losses = []
        for i in range(n_iterations):
            self.seed = i  # Different seed each iteration
            annual_loss = self.simulate_year(region, species, climate_scenario)
            losses.append(annual_loss)
            
        return np.array(losses)