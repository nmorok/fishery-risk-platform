import numpy as np
from scipy import stats

class LossModel:
    """
    Estimate financial loss from disaster characteristics
    Following Little et al. (2016) methodology
    """
    
    def __init__(self, model_type='log-linear'):
        """
        model_type: 'linear', 'log-log', or 'log-linear'
        """
        self.model_type = model_type
        self.params = None
        
    def fit(self, X, y):
        """
        Fit loss model to historical data
        
        Args:
            X: DataFrame with features (region, species, severity, duration)
            y: Array of loss amounts (USD)
        """
        pass
        
    def predict(self, X):
        """
        Predict loss for new disaster scenarios
        
        Returns:
            Array of predicted losses
        """
        pass

    loss = β₀ + β₁(cumulative_intensity) + β₂(peak_intensity) + 
       β₃(duration) + β₄(region) + β₅(species) + ε
    
    log(loss) = β₀ + β₁(cumulative_intensity) + β₂(peak_intensity) + 
            β₃(duration) + β₄(region) + β₅(species) + ε
    
    log(loss) = β₀ + β₁log(cumulative_intensity) + β₂log(peak_intensity) + 
            β₃log(duration+1) + β₄(region) + β₅(species) + ε
    
    def compare_models(disasters_df, heatwaves_df):
    """
    Fit all three models and compare via AIC, RMSE, R²
    
    Returns:
        DataFrame with model comparison metrics
    """
    pass