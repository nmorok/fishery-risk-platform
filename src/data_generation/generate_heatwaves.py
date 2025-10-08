class HeatwaveGenerator:
    """Generate marine heatwave severity metrics"""
    
    def generate_heatwaves(self, disaster_df):
        """
        For disasters where type == 'Warm ocean conditions',
        generate heatwave metrics
        
        Returns:
            pd.DataFrame with columns:
            - disaster_id (foreign key)
            - cumulative_intensity (degree-days)
            - peak_intensity (°C above baseline)
            - duration_days
            - max_extent_km2
        """
        pass