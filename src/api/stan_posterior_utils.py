"""
Full Bayesian Monte Carlo using fitted distributions for disaster characteristics.

This version:
1. Loads posterior samples from Stan
2. Loads fitted distributions for disaster characteristics
3. Samples from distributions (not historical data directly)
4. Runs Monte Carlo with proper uncertainty quantification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import json

# Paths
CSV_DIR = Path('data/csv')
OUTPUT_DIR = Path('data/output')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ==============================================================================
# Load data and fitted distributions
# ==============================================================================

print("="*80)
print("BAYESIAN MONTE CARLO WITH FITTED DISTRIBUTIONS")
print("="*80)

# Load original data (for disaster counts and scaling parameters)
df = pd.read_csv(CSV_DIR / 'model_data.csv')
df = df.dropna(subset=['log_appropriation', 'log_total_value', 'peak_intensity',
                       'duration_days', 'percent_in_heatwave'])
df = df[(df['peak_intensity'] > 0) & (df['log_appropriation'] > 0)]

# Load posterior samples
print("\nLoading posterior samples...")
posterior_samples = pd.read_csv(OUTPUT_DIR / 'posterior_samples.csv')
print(f"Loaded {len(posterior_samples)} posterior samples")

# Load fitted distributions
print("\nLoading fitted distributions...")
with open(OUTPUT_DIR / 'fitted_distributions.json', 'r') as f:
    fitted_distributions = json.load(f)

print("\nFitted distributions:")
for var, fit_info in fitted_distributions.items():
    print(f"  {var}: {fit_info['distribution']}")

# Load scaling parameters
scaling_params = pd.read_csv(OUTPUT_DIR / 'scaling_parameters.csv').iloc[0].to_dict()

print("\nScaling parameters loaded")

# ==============================================================================
# Sampling functions
# ==============================================================================

def sample_from_fitted_distribution(fit_info, n_samples=1):
    """
    Sample from a fitted distribution.
    
    Parameters:
    -----------
    fit_info : dict
        Dictionary with 'distribution', 'params', and optionally 'scale'
    n_samples : int
        Number of samples to draw
        
    Returns:
    --------
    array : Sampled values
    """
    dist_name = fit_info['distribution']
    params = fit_info['params']
    
    # Get distribution object
    dist = getattr(stats, dist_name)
    
    # Sample
    samples = dist.rvs(*params, size=n_samples)
    
    # Apply scale if needed (for percent_in_heatwave if using beta)
    if 'scale' in fit_info:
        samples = samples * fit_info['scale']
    
    # For percent_in_heatwave, ensure values are in [0, 100]
    if 'truncate_0_100' in fit_info and fit_info['truncate_0_100']:
        samples = np.clip(samples, 0, 100)
    
    return samples


def simulate_year_from_distributions(n_disasters, fitted_distributions, 
                                     scaling_params, posterior_samples, 
                                     sample_idx=None):
    """
    Simulate a year using fitted distributions (not bootstrap).
    
    Parameters:
    -----------
    n_disasters : int
        Number of disasters to simulate
    fitted_distributions : dict
        Fitted distribution parameters
    scaling_params : dict
        Standardization parameters
    posterior_samples : pd.DataFrame
        Full posterior samples from Stan
    sample_idx : int or None
        If None, randomly select a posterior sample
        
    Returns:
    --------
    tuple : (total_appropriation, individual_appropriations)
    """
    
    # Select a posterior sample
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(posterior_samples))
    
    posterior_draw = posterior_samples.iloc[sample_idx]
    
    # Extract parameters from this draw
    beta_0 = posterior_draw['alpha']
    beta_value = posterior_draw['beta_value']
    beta_peak = posterior_draw['beta_peak']
    beta_duration = posterior_draw['beta_duration']
    beta_percent = posterior_draw['beta_percent']
    sigma = posterior_draw['sigma']
    
    # Sample disaster characteristics from FITTED DISTRIBUTIONS
    log_values = sample_from_fitted_distribution(
        fitted_distributions['log_total_value'], n_disasters
    )
    peak_intensities = sample_from_fitted_distribution(
        fitted_distributions['peak_intensity'], n_disasters
    )
    durations = sample_from_fitted_distribution(
        fitted_distributions['duration_days'], n_disasters
    )
    
    # SAFETY CHECK: Cap extreme duration values to prevent numerical overflow
    # Use 99th percentile of historical data as upper bound
    max_reasonable_duration = 2_600_000  # Just above historical max of 2.57M
    durations = np.clip(durations, 0, max_reasonable_duration)
    
    percents = sample_from_fitted_distribution(
        fitted_distributions['percent_in_heatwave'], n_disasters
    )
    
    # Standardize predictors (using same scaling as model fitting)
    x_value = (log_values - scaling_params['log_value_mean']) / scaling_params['log_value_std']
    x_peak = (peak_intensities - scaling_params['peak_mean']) / scaling_params['peak_std']
    x_duration = (durations - scaling_params['duration_mean']) / scaling_params['duration_std']
    x_percent = (percents - scaling_params['percent_mean']) / scaling_params['percent_std']
    
    # Predict with this posterior draw
    log_appropriations = (
        beta_0 + 
        beta_value * x_value +
        beta_peak * x_peak +
        beta_duration * x_duration +
        beta_percent * x_percent +
        np.random.normal(0, sigma, size=n_disasters)
    )
    
    appropriations = np.exp(log_appropriations)
    
    # SAFETY CHECK: Cap at 10x the historical maximum
    # Historical max single appropriation was ~$252M
    max_reasonable_approp = 2_500_000_000  # $2.5B per disaster
    appropriations = np.clip(appropriations, 0, max_reasonable_approp)
    
    return appropriations.sum(), appropriations


def run_monte_carlo_with_distributions(posterior_samples, fitted_distributions,
                                       scaling_params, df,
                                       n_simulations=10000, 
                                       n_disasters_dist='empirical',
                                       seed=42):
    """
    Run Monte Carlo simulation using fitted distributions.
    
    Parameters:
    -----------
    posterior_samples : pd.DataFrame
        Posterior samples from Stan
    fitted_distributions : dict
        Fitted distribution parameters
    scaling_params : dict
        Standardization parameters
    df : pd.DataFrame
        Historical data (for disaster count distribution)
    n_simulations : int
        Number of years to simulate
    n_disasters_dist : str or int
        How to sample disaster counts
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame : Simulation results
    """
    np.random.seed(seed)
    
    disasters_per_year = df.groupby('sst_year').size()
    results = []
    
    print(f"\nRunning {n_simulations:,} simulations...")
    
    for sim in range(n_simulations):
        # Determine number of disasters
        if n_disasters_dist == 'empirical':
            n_disasters = np.random.choice(disasters_per_year.values)
        elif isinstance(n_disasters_dist, int):
            n_disasters = n_disasters_dist
        else:
            n_disasters = int(np.round(disasters_per_year.mean()))
        
        # Simulate using fitted distributions
        total_approp, individual_approps = simulate_year_from_distributions(
            n_disasters, fitted_distributions, scaling_params, posterior_samples
        )
        
        results.append({
            'simulation': sim,
            'n_disasters': n_disasters,
            'total_appropriation': total_approp,
            'mean_appropriation': total_approp / n_disasters if n_disasters > 0 else 0
        })
        
        if (sim + 1) % 1000 == 0:
            print(f"  Completed {sim + 1:,}/{n_simulations:,} simulations")
    
    return pd.DataFrame(results)


# ==============================================================================
# Climate change scenario function
# ==============================================================================

def create_climate_scenario(fitted_distributions, intensity_increase=1.2, 
                            duration_increase=1.15, coverage_increase=1.1):
    """
    Create modified distributions for climate change scenario.
    
    Parameters:
    -----------
    fitted_distributions : dict
        Original fitted distributions
    intensity_increase : float
        Multiplicative factor for peak intensity
    duration_increase : float
        Multiplicative factor for duration
    coverage_increase : float
        Multiplicative factor for spatial coverage
        
    Returns:
    --------
    dict : Modified distributions
    """
    
    import copy
    modified = copy.deepcopy(fitted_distributions)
    
    # For gamma/lognormal distributions, we can shift parameters
    # This is simplified - for publication you'd want to carefully adjust shape parameters
    
    # Peak intensity: increase the location/scale
    if modified['peak_intensity']['distribution'] == 'gamma':
        # For gamma(a, loc, scale), mean = a*scale + loc
        # Increase scale parameter
        params = list(modified['peak_intensity']['params'])
        params[2] = params[2] * intensity_increase  # scale parameter
        modified['peak_intensity']['params'] = params
        modified['peak_intensity']['mean'] *= intensity_increase
    
    # Duration: increase scale
    if modified['duration_days']['distribution'] == 'gamma':
        params = list(modified['duration_days']['params'])
        params[2] = params[2] * duration_increase
        modified['duration_days']['params'] = params
        modified['duration_days']['mean'] *= duration_increase
    
    # Percent coverage: shift distribution toward higher values
    # For beta, increase alpha parameter (shifts toward 1)
    if modified['percent_in_heatwave']['distribution'] == 'beta':
        params = list(modified['percent_in_heatwave']['params'])
        params[0] = params[0] * coverage_increase  # alpha parameter
        modified['percent_in_heatwave']['params'] = params
    
    return modified


# ==============================================================================
# Run simulations
# ==============================================================================

print("\n" + "="*80)
print("RUNNING BASELINE SIMULATION")
print("="*80)

baseline_results = run_monte_carlo_with_distributions(
    posterior_samples, fitted_distributions, scaling_params, df,
    n_simulations=10000
)

print("\n" + "="*80)
print("BASELINE RESULTS")
print("="*80)

mean_approp = baseline_results['total_appropriation'].mean()
median_approp = baseline_results['total_appropriation'].median()
ci_low = baseline_results['total_appropriation'].quantile(0.025)
ci_high = baseline_results['total_appropriation'].quantile(0.975)

print(f"\nYearly appropriation:")
print(f"  Mean: ${mean_approp/1e6:.1f}M")
print(f"  Median: ${median_approp/1e6:.1f}M")
print(f"  95% CI: ${ci_low/1e6:.1f}M - ${ci_high/1e6:.1f}M")

prob_100m = (baseline_results['total_appropriation'] > 100e6).mean()
prob_200m = (baseline_results['total_appropriation'] > 200e6).mean()
print(f"\nProbabilities:")
print(f"  P(> $100M) = {prob_100m:.1%}")
print(f"  P(> $200M) = {prob_200m:.1%}")

# Save baseline results
baseline_results.to_csv(OUTPUT_DIR / 'mc_with_distributions_baseline.csv', index=False)
print(f"\nSaved baseline results to {OUTPUT_DIR / 'mc_with_distributions_baseline.csv'}")

# ==============================================================================
# Climate change scenario
# ==============================================================================

print("\n" + "="*80)
print("RUNNING CLIMATE CHANGE SCENARIO (+20% intensity, +15% duration, +10% coverage)")
print("="*80)

climate_distributions = create_climate_scenario(
    fitted_distributions,
    intensity_increase=1.2,
    duration_increase=1.15,
    coverage_increase=1.1
)

climate_results = run_monte_carlo_with_distributions(
    posterior_samples, climate_distributions, scaling_params, df,
    n_simulations=10000, seed=43
)

print("\n" + "="*80)
print("CLIMATE SCENARIO RESULTS")
print("="*80)

mean_approp_climate = climate_results['total_appropriation'].mean()
median_approp_climate = climate_results['total_appropriation'].median()
ci_low_climate = climate_results['total_appropriation'].quantile(0.025)
ci_high_climate = climate_results['total_appropriation'].quantile(0.975)

print(f"\nYearly appropriation:")
print(f"  Mean: ${mean_approp_climate/1e6:.1f}M")
print(f"  Median: ${median_approp_climate/1e6:.1f}M")
print(f"  95% CI: ${ci_low_climate/1e6:.1f}M - ${ci_high_climate/1e6:.1f}M")

increase_pct = ((mean_approp_climate - mean_approp) / mean_approp) * 100
print(f"\nIncrease from baseline: +{increase_pct:.1f}%")

prob_100m_climate = (climate_results['total_appropriation'] > 100e6).mean()
prob_200m_climate = (climate_results['total_appropriation'] > 200e6).mean()
print(f"\nProbabilities:")
print(f"  P(> $100M) = {prob_100m_climate:.1%} (baseline: {prob_100m:.1%})")
print(f"  P(> $200M) = {prob_200m_climate:.1%} (baseline: {prob_200m:.1%})")

# Save climate results
climate_results.to_csv(OUTPUT_DIR / 'mc_with_distributions_climate.csv', index=False)
print(f"\nSaved climate results to {OUTPUT_DIR / 'mc_with_distributions_climate.csv'}")

# ==============================================================================
# Comparison visualization
# ==============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Distribution comparison
ax = axes[0, 0]
ax.hist(baseline_results['total_appropriation'] / 1e6, bins=50, alpha=0.6,
        label='Baseline', edgecolor='black', color='steelblue', density=True)
ax.hist(climate_results['total_appropriation'] / 1e6, bins=50, alpha=0.6,
        label='Climate (+20%)', edgecolor='black', color='red', density=True)
ax.axvline(mean_approp / 1e6, color='steelblue', linestyle='--', linewidth=2)
ax.axvline(mean_approp_climate / 1e6, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Total Yearly Appropriation ($ Millions)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('(A) Baseline vs. Climate Change Scenario', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel 2: Cumulative distributions
ax = axes[0, 1]
sorted_baseline = np.sort(baseline_results['total_appropriation'] / 1e6)
sorted_climate = np.sort(climate_results['total_appropriation'] / 1e6)
cumulative = np.arange(1, len(sorted_baseline) + 1) / len(sorted_baseline)

ax.plot(sorted_baseline, cumulative, linewidth=2, label='Baseline', color='steelblue')
ax.plot(sorted_climate, cumulative, linewidth=2, label='Climate (+20%)', color='red')
ax.set_xlabel('Total Yearly Appropriation ($ Millions)', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('(B) Cumulative Distribution Functions', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: Probability of exceeding thresholds
ax = axes[1, 0]
thresholds = np.arange(0, 500, 10)
probs_baseline = [(baseline_results['total_appropriation'] > t*1e6).mean() for t in thresholds]
probs_climate = [(climate_results['total_appropriation'] > t*1e6).mean() for t in thresholds]

ax.plot(thresholds, probs_baseline, linewidth=2, label='Baseline', color='steelblue')
ax.plot(thresholds, probs_climate, linewidth=2, label='Climate (+20%)', color='red')
ax.set_xlabel('Appropriation Threshold ($ Millions)', fontsize=12)
ax.set_ylabel('Probability of Exceeding', fontsize=12)
ax.set_title('(C) Exceedance Probabilities', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel 4: Difference in means by disaster count
ax = axes[1, 1]
baseline_by_n = baseline_results.groupby('n_disasters')['total_appropriation'].mean() / 1e6
climate_by_n = climate_results.groupby('n_disasters')['total_appropriation'].mean() / 1e6

n_disasters_range = sorted(set(baseline_results['n_disasters'].unique()) & 
                           set(climate_results['n_disasters'].unique()))

baseline_means = [baseline_by_n.get(n, 0) for n in n_disasters_range]
climate_means = [climate_by_n.get(n, 0) for n in n_disasters_range]

x = np.arange(len(n_disasters_range))
width = 0.35
ax.bar(x - width/2, baseline_means, width, label='Baseline', color='steelblue', alpha=0.7)
ax.bar(x + width/2, climate_means, width, label='Climate (+20%)', color='red', alpha=0.7)

ax.set_xlabel('Number of Disasters per Year', fontsize=12)
ax.set_ylabel('Mean Appropriation ($ Millions)', fontsize=12)
ax.set_title('(D) Mean Appropriation by Disaster Count', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(n_disasters_range)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '../figures/mc_distributions_comparison.png', 
            dpi=300, bbox_inches='tight')
print(f"Saved visualization to figures/mc_distributions_comparison.png")

# ==============================================================================
# Summary statistics
# ==============================================================================

summary_stats = pd.DataFrame({
    'Scenario': ['Baseline', 'Climate (+20%)'],
    'Mean (M)': [mean_approp/1e6, mean_approp_climate/1e6],
    'Median (M)': [median_approp/1e6, median_approp_climate/1e6],
    'CI_lower (M)': [ci_low/1e6, ci_low_climate/1e6],
    'CI_upper (M)': [ci_high/1e6, ci_high_climate/1e6],
    'P(>$100M)': [prob_100m, prob_100m_climate],
    'P(>$200M)': [prob_200m, prob_200m_climate]
})

summary_stats.to_csv(OUTPUT_DIR / 'mc_distributions_summary.csv', index=False)

print("\n" + "="*80)
print("SIMULATION COMPLETE")
print("="*80)

print("\nFiles saved:")
print(f"  - {OUTPUT_DIR / 'mc_with_distributions_baseline.csv'}")
print(f"  - {OUTPUT_DIR / 'mc_with_distributions_climate.csv'}")
print(f"  - {OUTPUT_DIR / 'mc_distributions_summary.csv'}")
print(f"  - figures/mc_distributions_comparison.png")

print("\n" + "="*80)