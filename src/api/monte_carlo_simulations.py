"""
Monte Carlo simulation for yearly disaster appropriations.

This script:
1. Loads the fitted Bayesian model parameters
2. Extracts historical distributions for disaster characteristics
3. Simulates years with random numbers of disasters
4. Predicts appropriations using the Bayesian model
5. Aggregates to yearly totals
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Paths
CSV_DIR = Path('data/csv')
OUTPUT_DIR = Path('data/output')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Load the data and model summary
df = pd.read_csv(CSV_DIR / 'model_data.csv')
model_summary = pd.read_csv('data/csv/bayesian_model_summary.csv', index_col=0)

# Clean data (same as in model fitting)
df = df.dropna(subset=[
    'log_appropriation',
    'log_total_value',
    'peak_intensity',
    'duration_days',
    'percent_in_heatwave'
])
df = df[(df['peak_intensity'] > 0) & (df['log_appropriation'] > 0)]

print(f"Using {len(df)} disasters for simulation")
print(f"\nHistorical data spans {df['sst_year'].min()} to {df['sst_year'].max()}")

# ==============================================================================
# 1. Extract fitted model parameters
# ==============================================================================
print("\n" + "="*80)
print("EXTRACTING MODEL PARAMETERS")
print("="*80)

# Get posterior means for coefficients
beta_0 = model_summary.loc['alpha', 'Mean']  # Note: Stan uses 'alpha' for intercept
beta_value = model_summary.loc['beta_value', 'Mean']
beta_peak = model_summary.loc['beta_peak', 'Mean']
beta_duration = model_summary.loc['beta_duration', 'Mean']
beta_percent = model_summary.loc['beta_percent', 'Mean']
sigma = model_summary.loc['sigma', 'Mean']

print(f"\nPosterior mean coefficients:")
print(f"  Intercept: {beta_0:.4f}")
print(f"  log(value): {beta_value:.4f}")
print(f"  Peak intensity: {beta_peak:.4f}")
print(f"  Duration: {beta_duration:.4f}")
print(f"  Percent in heatwave: {beta_percent:.4f}")
print(f"  Sigma: {sigma:.4f}")

# Store scaling parameters (used in model fitting)
scaling_params = {
    'log_value_mean': df['log_total_value'].mean(),
    'log_value_std': df['log_total_value'].std(),
    'peak_mean': df['peak_intensity'].mean(),
    'peak_std': df['peak_intensity'].std(),
    'duration_mean': df['duration_days'].mean(),
    'duration_std': df['duration_days'].std(),
    'percent_mean': df['percent_in_heatwave'].mean(),
    'percent_std': df['percent_in_heatwave'].std()
}

# ==============================================================================
# 2. Characterize historical distributions
# ==============================================================================
print("\n" + "="*80)
print("CHARACTERIZING HISTORICAL DISTRIBUTIONS")
print("="*80)

# Number of disasters per year
disasters_per_year = df.groupby('sst_year').size()
print(f"\nDisasters per year:")
print(f"  Mean: {disasters_per_year.mean():.2f}")
print(f"  Std: {disasters_per_year.std():.2f}")
print(f"  Min: {disasters_per_year.min()}")
print(f"  Max: {disasters_per_year.max()}")

# Distribution of predictor variables
print(f"\nPredictor distributions:")
for var in ['log_total_value', 'peak_intensity', 'duration_days', 'percent_in_heatwave']:
    print(f"  {var}:")
    print(f"    Mean: {df[var].mean():.2f}")
    print(f"    Std: {df[var].std():.2f}")
    print(f"    Min: {df[var].min():.2f}")
    print(f"    Max: {df[var].max():.2f}")

# ==============================================================================
# 3. Monte Carlo simulation function
# ==============================================================================

def simulate_year(n_disasters, df, scaling_params, beta_0, beta_value, 
                  beta_peak, beta_duration, beta_percent, sigma,
                  use_uncertainty=True):
    """
    Simulate appropriations for a single year with n_disasters.
    
    Parameters:
    -----------
    n_disasters : int
        Number of disasters to simulate
    df : pd.DataFrame
        Historical data to sample distributions from
    scaling_params : dict
        Mean and std for standardization
    beta_* : float
        Model coefficients
    sigma : float
        Model error standard deviation
    use_uncertainty : bool
        If True, add residual uncertainty to predictions
    
    Returns:
    --------
    float : Total appropriation for the year (in original dollars)
    list : Individual disaster appropriations
    """
    
    # Draw disaster characteristics from historical distributions
    # Using bootstrap resampling approach
    sampled_indices = np.random.choice(len(df), size=n_disasters, replace=True)
    sampled_disasters = df.iloc[sampled_indices]
    
    # Get raw values
    log_values = sampled_disasters['log_total_value'].values
    peak_intensities = sampled_disasters['peak_intensity'].values
    durations = sampled_disasters['duration_days'].values
    percents = sampled_disasters['percent_in_heatwave'].values
    
    # Standardize using the same scaling as model fitting
    x_value = (log_values - scaling_params['log_value_mean']) / scaling_params['log_value_std']
    x_peak = (peak_intensities - scaling_params['peak_mean']) / scaling_params['peak_std']
    x_duration = (durations - scaling_params['duration_mean']) / scaling_params['duration_std']
    x_percent = (percents - scaling_params['percent_mean']) / scaling_params['percent_std']
    
    # Predict log_appropriation
    log_appropriations = (
        beta_0 + 
        beta_value * x_value +
        beta_peak * x_peak +
        beta_duration * x_duration +
        beta_percent * x_percent
    )
    
    # Add uncertainty if requested
    if use_uncertainty:
        log_appropriations += np.random.normal(0, sigma, size=n_disasters)
    
    # Convert back to dollars
    appropriations = np.exp(log_appropriations)
    
    # Total for the year
    total_appropriation = appropriations.sum()
    
    return total_appropriation, appropriations.tolist()


def run_monte_carlo(n_simulations=10000, 
                   n_disasters_dist='empirical',
                   use_uncertainty=True,
                   seed=42):
    """
    Run Monte Carlo simulation of yearly appropriations.
    
    Parameters:
    -----------
    n_simulations : int
        Number of years to simulate
    n_disasters_dist : str or int
        If 'empirical': draw from historical distribution
        If 'mean': use mean of historical distribution
        If int: use fixed number
        If 'poisson': use Poisson with lambda = historical mean
    use_uncertainty : bool
        Whether to include model uncertainty in predictions
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame : Results with yearly totals and disaster counts
    """
    
    np.random.seed(seed)
    
    results = []
    
    for sim in range(n_simulations):
        # Determine number of disasters for this year
        if n_disasters_dist == 'empirical':
            n_disasters = np.random.choice(disasters_per_year.values)
        elif n_disasters_dist == 'mean':
            n_disasters = int(np.round(disasters_per_year.mean()))
        elif n_disasters_dist == 'poisson':
            n_disasters = np.random.poisson(disasters_per_year.mean())
        elif isinstance(n_disasters_dist, int):
            n_disasters = n_disasters_dist
        else:
            raise ValueError(f"Unknown distribution: {n_disasters_dist}")
        
        # Simulate the year
        total_approp, individual_approps = simulate_year(
            n_disasters, df, scaling_params,
            beta_0, beta_value, beta_peak, beta_duration, beta_percent, sigma,
            use_uncertainty=use_uncertainty
        )
        
        results.append({
            'simulation': sim,
            'n_disasters': n_disasters,
            'total_appropriation': total_approp,
            'mean_appropriation': total_approp / n_disasters if n_disasters > 0 else 0,
            'individual_appropriations': individual_approps
        })
    
    return pd.DataFrame(results)


# ==============================================================================
# 4. Run simulations
# ==============================================================================
print("\n" + "="*80)
print("RUNNING MONTE CARLO SIMULATIONS")
print("="*80)

# Run with different scenarios
scenarios = {
    'baseline': {'n_disasters_dist': 'empirical', 'use_uncertainty': True},
    'mean_disasters': {'n_disasters_dist': 'mean', 'use_uncertainty': True},
    'no_uncertainty': {'n_disasters_dist': 'empirical', 'use_uncertainty': False},
    'poisson_disasters': {'n_disasters_dist': 'poisson', 'use_uncertainty': True}
}

all_results = {}

for scenario_name, params in scenarios.items():
    print(f"\nRunning scenario: {scenario_name}")
    results_df = run_monte_carlo(n_simulations=10000, **params)
    all_results[scenario_name] = results_df
    
    print(f"  Mean yearly appropriation: ${results_df['total_appropriation'].mean():,.0f}")
    print(f"  Median yearly appropriation: ${results_df['total_appropriation'].median():,.0f}")
    print(f"  90% CI: ${results_df['total_appropriation'].quantile(0.05):,.0f} - "
          f"${results_df['total_appropriation'].quantile(0.95):,.0f}")
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / f'monte_carlo_{scenario_name}.csv', index=False)

# ==============================================================================
# 5. Visualization
# ==============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Distribution of yearly appropriations (baseline)
ax = axes[0, 0]
baseline_results = all_results['baseline']
ax.hist(baseline_results['total_appropriation'] / 1e6, bins=50, 
        alpha=0.7, edgecolor='black')
ax.axvline(baseline_results['total_appropriation'].mean() / 1e6, 
           color='red', linestyle='--', linewidth=2, label='Mean')
ax.axvline(baseline_results['total_appropriation'].median() / 1e6,
           color='orange', linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('Total Yearly Appropriation ($ Millions)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Simulated Yearly Appropriations\n(Baseline Scenario)', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Comparison across scenarios
ax = axes[0, 1]
scenario_data = [all_results[s]['total_appropriation'] / 1e6 for s in scenarios.keys()]
scenario_labels = list(scenarios.keys())
bp = ax.boxplot(scenario_data, labels=scenario_labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Total Yearly Appropriation ($ Millions)', fontsize=12)
ax.set_title('Comparison Across Scenarios', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Number of disasters vs total appropriation
ax = axes[1, 0]
ax.scatter(baseline_results['n_disasters'], 
           baseline_results['total_appropriation'] / 1e6,
           alpha=0.3, s=20)
ax.set_xlabel('Number of Disasters', fontsize=12)
ax.set_ylabel('Total Yearly Appropriation ($ Millions)', fontsize=12)
ax.set_title('Disasters per Year vs Total Appropriation', 
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Add trend line
z = np.polyfit(baseline_results['n_disasters'], 
               baseline_results['total_appropriation'] / 1e6, 1)
p = np.poly1d(z)
x_line = np.linspace(baseline_results['n_disasters'].min(), 
                     baseline_results['n_disasters'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend: y = {z[0]:.1f}x + {z[1]:.1f}')
ax.legend()

# Plot 4: Historical comparison
ax = axes[1, 1]
historical_yearly = df.groupby('sst_year')['appropriation'].sum() / 1e6
ax.hist(historical_yearly, bins=15, alpha=0.5, label='Historical', 
        edgecolor='black', color='green')
ax.hist(baseline_results['total_appropriation'] / 1e6, bins=50, 
        alpha=0.5, label='Simulated', edgecolor='black', color='blue')
ax.set_xlabel('Total Yearly Appropriation ($ Millions)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Simulated vs Historical Appropriations', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'monte_carlo_results.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to {OUTPUT_DIR / 'monte_carlo_results.png'}")

# ==============================================================================
# 6. Summary statistics
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary_stats = []
for scenario_name, results_df in all_results.items():
    stats_dict = {
        'scenario': scenario_name,
        'mean_yearly': results_df['total_appropriation'].mean(),
        'median_yearly': results_df['total_appropriation'].median(),
        'std_yearly': results_df['total_appropriation'].std(),
        'q05': results_df['total_appropriation'].quantile(0.05),
        'q95': results_df['total_appropriation'].quantile(0.95),
        'mean_disasters': results_df['n_disasters'].mean(),
        'mean_per_disaster': results_df['mean_appropriation'].mean()
    }
    summary_stats.append(stats_dict)

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(OUTPUT_DIR / 'monte_carlo_summary.csv', index=False)

print("\n" + summary_df.to_string())

print("\n" + "="*80)
print("MONTE CARLO SIMULATION COMPLETE")
print("="*80)
print(f"\nResults saved to {OUTPUT_DIR}/")
print("  - monte_carlo_baseline.csv")
print("  - monte_carlo_mean_disasters.csv")
print("  - monte_carlo_no_uncertainty.csv")
print("  - monte_carlo_poisson_disasters.csv")
print("  - monte_carlo_summary.csv")
print("  - monte_carlo_results.png")