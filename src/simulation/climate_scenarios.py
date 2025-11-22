CLIMATE_SCENARIOS = {
    'baseline': {
        'frequency_multiplier': 1.0,
        'intensity_multiplier': 1.0,
        'duration_multiplier': 1.0
    },
    'moderate': {
        'frequency_multiplier': 1.34,  # +34% from Oliver et al.
        'intensity_multiplier': 1.50,  # +50%
        'duration_multiplier': 1.17    # +17%
    },
    'severe': {
        'frequency_multiplier': 1.45,  # +45%
        'intensity_multiplier': 1.75,  # +75%
        'duration_multiplier': 1.225   # +22.5%
    }
}

def adjust_for_climate(base_value, scenario, parameter):
    """Apply climate scenario adjustments"""
    multiplier = CLIMATE_SCENARIOS[scenario][f'{parameter}_multiplier']
    return base_value * multiplier