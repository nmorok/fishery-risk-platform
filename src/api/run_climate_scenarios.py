"""
Climate Change Scenario Analysis for Fishery Disaster Appropriations.

This script creates and compares three scenarios:
- Baseline: Current historical conditions
- Moderate Climate Change: +20% frequency, peak, duration, coverage
- High Climate Change: +40% frequency, peak, duration, coverage

Runs full Bayesian Monte Carlo for each scenario and compares results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import copy
from scipy.special import factorial

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Paths
CSV_DIR = Path('data/csv')
OUTPUT_DIR = Path('data/output')
FIGURES_DIR = Path('figures/climate_scenarios')
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("CLIMATE CHANGE SCENARIO ANALYSIS")
print("="*80)

# ==============================================================================
# Load data and fitted distributions
# ==============================================================================

print("\nLoading data and distributions...")
df = pd.read_csv(CSV_DIR / 'model_data.csv')
df = df.dropna(subset=['log_appropriation', 'log_total_value', 'peak_intensity',
                       'duration_days', 'percent_in_heatwave'])
df = df[(df['peak_intensity'] > 0) & (df['log_appropriation'] > 0)]

posterior_samples = pd.read_csv(OUTPUT_DIR / 'posterior_samples.csv')
with open(OUTPUT_DIR / 'fitted_distributions.json', 'r') as f:
    baseline_distributions = json.load(f)
scaling_params = pd.read_csv(OUTPUT_DIR / 'scaling_parameters.csv').iloc[0].to_dict()

print(f"Loaded {len(posterior_samples)} posterior samples")
print(f"Using {len(df)} historical disasters")

# ==============================================================================
# Climate scenario creation functions
# ==============================================================================

def create_climate_scenario(fitted_distributions, 
                            frequency_increase=1.0,
                            intensity_increase=1.0, 
                            duration_increase=1.0, 
                            coverage_increase=1.0,
                            scenario_name="Custom"):
    """
    Create modified distributions for climate change scenario.
    
    Parameters:
    -----------
    fitted_distributions : dict
        Original fitted distributions
    frequency_increase : float
        Multiplier for disaster frequency (e.g., 1.2 = 20% increase)
    intensity_increase : float
        Multiplier for peak intensity
    duration_increase : float
        Multiplier for duration
    coverage_increase : float
        Multiplier for spatial coverage
    scenario_name : str
        Name of scenario for reporting
        
    Returns:
    --------
    dict : Modified distributions
    dict : Scenario metadata
    """
    
    modified = copy.deepcopy(fitted_distributions)
    
    print(f"\n{scenario_name} Scenario:")
    print(f"  Frequency: +{(frequency_increase-1)*100:.0f}%")
    print(f"  Intensity: +{(intensity_increase-1)*100:.0f}%")
    print(f"  Duration: +{(duration_increase-1)*100:.0f}%")
    print(f"  Coverage: +{(coverage_increase-1)*100:.0f}%")
    
    # Modify peak intensity
    if modified['peak_intensity']['distribution'] == 'weibull_min':
        # For Weibull: shift location parameter
        params = list(modified['peak_intensity']['params'])
        original_loc = params[1]
        params[1] = params[1] + (intensity_increase - 1.0) * modified['peak_intensity']['mean']
        modified['peak_intensity']['params'] = params
        
        # Recalculate mean
        dist = stats.weibull_min
        modified['peak_intensity']['mean'] = dist.mean(*params)
        print(f"    Peak intensity mean: {baseline_distributions['peak_intensity']['mean']:.2f} → "
              f"{modified['peak_intensity']['mean']:.2f}")
    
    elif modified['peak_intensity']['distribution'] == 'gamma':
        # For gamma: increase scale parameter
        params = list(modified['peak_intensity']['params'])
        params[2] = params[2] * intensity_increase  # scale parameter
        modified['peak_intensity']['params'] = params
        modified['peak_intensity']['mean'] *= intensity_increase
        print(f"    Peak intensity mean: {baseline_distributions['peak_intensity']['mean']:.2f} → "
              f"{modified['peak_intensity']['mean']:.2f}")
    
    # Modify duration
    if modified['duration_days']['distribution'] == 'gamma':
        params = list(modified['duration_days']['params'])
        params[2] = params[2] * duration_increase  # scale parameter
        modified['duration_days']['params'] = params
        modified['duration_days']['mean'] *= duration_increase
        print(f"    Duration mean: {baseline_distributions['duration_days']['mean']/1e6:.2f}M → "
              f"{modified['duration_days']['mean']/1e6:.2f}M days")
    
    elif modified['duration_days']['distribution'] == 'lognorm':
        # For lognormal: increase the scale parameter
        params = list(modified['duration_days']['params'])
        params[2] = params[2] * duration_increase
        modified['duration_days']['params'] = params
        modified['duration_days']['mean'] *= duration_increase
        print(f"    Duration mean: {baseline_distributions['duration_days']['mean']/1e6:.2f}M → "
              f"{modified['duration_days']['mean']/1e6:.2f}M days")
    
    # Modify spatial coverage (percent_in_heatwave)
    if modified['percent_in_heatwave']['distribution'] == 'beta':
        # For beta distribution, increase alpha parameter to shift toward higher values
        params = list(modified['percent_in_heatwave']['params'])
        # Increase alpha (first parameter) to shift distribution right
        params[0] = params[0] * (1 + (coverage_increase - 1.0))
        modified['percent_in_heatwave']['params'] = params
        
        # Recalculate mean
        dist = stats.beta
        if 'scale' in modified['percent_in_heatwave']:
            modified['percent_in_heatwave']['mean'] = dist.mean(*params) * 100
        else:
            modified['percent_in_heatwave']['mean'] = dist.mean(*params)
        print(f"    Coverage mean: {baseline_distributions['percent_in_heatwave']['mean']:.1f}% → "
              f"{modified['percent_in_heatwave']['mean']:.1f}%")
    
    # Metadata
    scenario_metadata = {
        'name': scenario_name,
        'frequency_multiplier': frequency_increase,
        'intensity_multiplier': intensity_increase,
        'duration_multiplier': duration_increase,
        'coverage_multiplier': coverage_increase
    }
    
    return modified, scenario_metadata


# ==============================================================================
# Monte Carlo sampling functions
# ==============================================================================

def sample_from_fitted_distribution(fit_info, n_samples=1):
    """Sample from fitted distribution with safety checks."""
    dist_name = fit_info['distribution']
    params = fit_info['params']
    dist = getattr(stats, dist_name)
    samples = dist.rvs(*params, size=n_samples)
    
    if 'scale' in fit_info:
        samples = samples * fit_info['scale']
    if 'truncate_0_100' in fit_info and fit_info['truncate_0_100']:
        samples = np.clip(samples, 0, 100)
    
    return samples


def simulate_year_climate_scenario(n_disasters, fitted_distributions, 
                                   scaling_params, posterior_samples):
    """Simulate a single year using climate-modified distributions."""
    
    # Random posterior sample
    sample_idx = np.random.randint(0, len(posterior_samples))
    posterior_draw = posterior_samples.iloc[sample_idx]
    
    # Extract parameters
    beta_0 = posterior_draw['alpha']
    beta_value = posterior_draw['beta_value']
    beta_peak = posterior_draw['beta_peak']
    beta_duration = posterior_draw['beta_duration']
    beta_percent = posterior_draw['beta_percent']
    sigma = posterior_draw['sigma']
    
    # Sample disaster characteristics from modified distributions
    log_values = sample_from_fitted_distribution(
        fitted_distributions['log_total_value'], n_disasters
    )
    peak_intensities = sample_from_fitted_distribution(
        fitted_distributions['peak_intensity'], n_disasters
    )
    durations = sample_from_fitted_distribution(
        fitted_distributions['duration_days'], n_disasters
    )
    percents = sample_from_fitted_distribution(
        fitted_distributions['percent_in_heatwave'], n_disasters
    )
    
    # Safety caps
    max_reasonable_duration = 2_600_000
    durations = np.clip(durations, 0, max_reasonable_duration)
    
    # Standardize predictors
    x_value = (log_values - scaling_params['log_value_mean']) / scaling_params['log_value_std']
    x_peak = (peak_intensities - scaling_params['peak_mean']) / scaling_params['peak_std']
    x_duration = (durations - scaling_params['duration_mean']) / scaling_params['duration_std']
    x_percent = (percents - scaling_params['percent_mean']) / scaling_params['percent_std']
    
    # Predict
    log_appropriations = (
        beta_0 + 
        beta_value * x_value +
        beta_peak * x_peak +
        beta_duration * x_duration +
        beta_percent * x_percent +
        np.random.normal(0, sigma, size=n_disasters)
    )
    
    appropriations = np.exp(log_appropriations)
    
    # Safety cap
    max_reasonable_approp = 2_500_000_000
    appropriations = np.clip(appropriations, 0, max_reasonable_approp)
    
    return appropriations.sum(), appropriations


def run_scenario_monte_carlo(scenario_name, fitted_distributions, 
                             frequency_multiplier, n_simulations=10000):
    """Run Monte Carlo for a climate scenario."""
    
    print(f"\nRunning {n_simulations:,} simulations for {scenario_name}...")
    
    # Load disaster frequency distribution
    disaster_freq_fit = fitted_distributions.get('disaster_frequency', None)
    
    if disaster_freq_fit is None:
        # Fallback to empirical if not fitted
        disasters_per_year = df.groupby('sst_year').size()
        print("  Warning: Using empirical disaster frequency (run fit_disaster_frequency.py first)")
        use_fitted_frequency = False
    else:
        use_fitted_frequency = True
        dist_name = disaster_freq_fit['distribution']
        params = disaster_freq_fit['params']
        print(f"  Using fitted {dist_name} distribution for disaster frequency")
    
    results = []
    
    for sim in range(n_simulations):
        # Sample disaster count (ZERO-TRUNCATED)
        if use_fitted_frequency:
            if dist_name == 'zt_poisson':
                # Zero-truncated Poisson: sample from regular Poisson then reject zeros
                base_lambda = params['lambda'] * frequency_multiplier
                n_disasters = 0
                while n_disasters == 0:  # Ensure at least 1 disaster
                    n_disasters = np.random.poisson(base_lambda)
            elif dist_name == 'zt_nbinom':
                # Zero-truncated Negative Binomial
                n_orig = params['n']
                p_orig = params['p']
                mean_orig = n_orig * (1 - p_orig) / p_orig
                mean_new = mean_orig * frequency_multiplier
                p_new = n_orig / (n_orig + mean_new)
                n_disasters = 0
                while n_disasters == 0:  # Ensure at least 1 disaster
                    n_disasters = np.random.negative_binomial(n_orig, p_new)
            else:  # empirical fallback
                empirical_data = np.array(params['data'])
                base_n_disasters = np.random.choice(empirical_data)
                n_disasters = int(np.round(base_n_disasters * frequency_multiplier))
        else:
            # Empirical sampling
            base_n_disasters = np.random.choice(disasters_per_year.values)
            n_disasters = int(np.round(base_n_disasters * frequency_multiplier))
        
        n_disasters = max(1, n_disasters)  # Ensure at least 1 disaster
        
        # Simulate year
        total_approp, individual_approps = simulate_year_climate_scenario(
            n_disasters, fitted_distributions, scaling_params, posterior_samples
        )
        
        results.append({
            'simulation': sim,
            'n_disasters': n_disasters,
            'total_appropriation': total_approp,
            'mean_appropriation': total_approp / n_disasters if n_disasters > 0 else 0
        })
        
        if (sim + 1) % 2000 == 0:
            print(f"  Completed {sim + 1:,}/{n_simulations:,}")
    
    return pd.DataFrame(results)


# ==============================================================================
# Run all three scenarios
# ==============================================================================

print("\n" + "="*80)
print("DEFINING CLIMATE SCENARIOS")
print("="*80)

# Baseline
baseline_metadata = {
    'name': 'Baseline',
    'frequency_multiplier': 1.0,
    'intensity_multiplier': 1.0,
    'duration_multiplier': 1.0,
    'coverage_multiplier': 1.0
}

# Moderate scenario (+20%)
moderate_distributions, moderate_metadata = create_climate_scenario(
    baseline_distributions,
    frequency_increase=1.20,
    intensity_increase=1.20,
    duration_increase=1.20,
    coverage_increase=1.20,
    scenario_name="Moderate (+20%)"
)

# High scenario (+40%)
high_distributions, high_metadata = create_climate_scenario(
    baseline_distributions,
    frequency_increase=1.40,
    intensity_increase=1.40,
    duration_increase=1.40,
    coverage_increase=1.40,
    scenario_name="High (+40%)"
)

# ==============================================================================
# Run Monte Carlo for all scenarios
# ==============================================================================

print("\n" + "="*80)
print("RUNNING MONTE CARLO SIMULATIONS")
print("="*80)

# Baseline
print("\n1. BASELINE SCENARIO")
baseline_results = run_scenario_monte_carlo(
    "Baseline", baseline_distributions, 1.0, n_simulations=10000
)

# Moderate
print("\n2. MODERATE CLIMATE SCENARIO (+20%)")
moderate_results = run_scenario_monte_carlo(
    "Moderate", moderate_distributions, 1.20, n_simulations=10000
)

# High
print("\n3. HIGH CLIMATE SCENARIO (+40%)")
high_results = run_scenario_monte_carlo(
    "High", high_distributions, 1.40, n_simulations=10000
)

# ==============================================================================
# Save results
# ==============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

baseline_results.to_csv(OUTPUT_DIR / 'climate_scenario_baseline.csv', index=False)
moderate_results.to_csv(OUTPUT_DIR / 'climate_scenario_moderate.csv', index=False)
high_results.to_csv(OUTPUT_DIR / 'climate_scenario_high.csv', index=False)

print(f"\nSaved simulation results to {OUTPUT_DIR}/")

# ==============================================================================
# Compare scenarios
# ==============================================================================

print("\n" + "="*80)
print("SCENARIO COMPARISON")
print("="*80)

scenarios = {
    'Baseline': baseline_results,
    'Moderate (+20%)': moderate_results,
    'High (+40%)': high_results
}

comparison_data = []

for scenario_name, results in scenarios.items():
    mean_approp = results['total_appropriation'].mean()
    median_approp = results['total_appropriation'].median()
    ci_low = results['total_appropriation'].quantile(0.025)
    ci_high = results['total_appropriation'].quantile(0.975)
    mean_disasters = results['n_disasters'].mean()
    prob_100m = (results['total_appropriation'] > 100e6).mean()
    prob_200m = (results['total_appropriation'] > 200e6).mean()
    
    comparison_data.append({
        'Scenario': scenario_name,
        'Mean Disasters/Year': mean_disasters,
        'Mean Appropriation ($M)': mean_approp / 1e6,
        'Median Appropriation ($M)': median_approp / 1e6,
        '95% CI Lower ($M)': ci_low / 1e6,
        '95% CI Upper ($M)': ci_high / 1e6,
        'P(>$100M)': prob_100m,
        'P(>$200M)': prob_200m
    })
    
    print(f"\n{scenario_name}:")
    print(f"  Mean disasters/year: {mean_disasters:.2f}")
    print(f"  Mean appropriation: ${mean_approp/1e6:.1f}M")
    print(f"  Median appropriation: ${median_approp/1e6:.1f}M")
    print(f"  95% CI: ${ci_low/1e6:.1f}M - ${ci_high/1e6:.1f}M")
    print(f"  P(>$100M): {prob_100m:.1%}")
    print(f"  P(>$200M): {prob_200m:.1%}")

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(OUTPUT_DIR / 'climate_scenario_comparison.csv', index=False)

# Calculate percent increases
baseline_mean = comparison_data[0]['Mean Appropriation ($M)']
moderate_increase = ((comparison_data[1]['Mean Appropriation ($M)'] - baseline_mean) / baseline_mean) * 100
high_increase = ((comparison_data[2]['Mean Appropriation ($M)'] - baseline_mean) / baseline_mean) * 100

print(f"\n" + "="*80)
print("CLIMATE IMPACT")
print("="*80)
print(f"\nModerate scenario (+20% climate): +{moderate_increase:.1f}% appropriations")
print(f"High scenario (+40% climate): +{high_increase:.1f}% appropriations")

# ==============================================================================
# Visualizations
# ==============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Figure 1: Three-way comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Distribution comparison
ax = axes[0, 0]
for scenario_name, results in scenarios.items():
    color = {'Baseline': 'steelblue', 'Moderate (+20%)': 'orange', 'High (+40%)': 'red'}[scenario_name]
    ax.hist(results['total_appropriation'] / 1e6, bins=50, alpha=0.5,
           label=scenario_name, edgecolor='black', color=color, density=True)

ax.set_xlabel('Total Yearly Appropriation ($ Millions)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(A) Distribution Comparison', fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)
ax.set_xlim(0, 500)

# Panel B: Cumulative distributions
ax = axes[0, 1]
for scenario_name, results in scenarios.items():
    color = {'Baseline': 'steelblue', 'Moderate (+20%)': 'orange', 'High (+40%)': 'red'}[scenario_name]
    sorted_vals = np.sort(results['total_appropriation'] / 1e6)
    cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cumulative, linewidth=2.5, label=scenario_name, color=color)

ax.set_xlabel('Total Yearly Appropriation ($ Millions)', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('(B) Cumulative Distribution Functions', fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)
ax.set_xlim(0, 500)

# Panel D: Mean appropriations by scenario
ax = axes[1, 0]
scenario_names = [d['Scenario'] for d in comparison_data]
means = [d['Mean Appropriation ($M)'] for d in comparison_data]
ci_lows = [d['95% CI Lower ($M)'] for d in comparison_data]
ci_highs = [d['95% CI Upper ($M)'] for d in comparison_data]
errors_low = [means[i] - ci_lows[i] for i in range(3)]
errors_high = [ci_highs[i] - means[i] for i in range(3)]

x_pos = np.arange(len(scenario_names))
colors_bar = ['steelblue', 'orange', 'red']
ax.bar(x_pos, means, color=colors_bar, alpha=0.7, edgecolor='black')
ax.errorbar(x_pos, means, yerr=[errors_low, errors_high], fmt='none', 
           color='black', capsize=5, capthick=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(scenario_names)
ax.set_ylabel('Mean Yearly Appropriation ($ Millions)', fontsize=11)
ax.set_title('(C) Mean Appropriations with 95% CI', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

# Panel D: Probability comparison
ax = axes[1, 1]
thresholds = np.arange(0, 400, 10)
for scenario_name, results in scenarios.items():
    color = {'Baseline': 'steelblue', 'Moderate (+20%)': 'orange', 'High (+40%)': 'red'}[scenario_name]
    probs = [(results['total_appropriation'] > t*1e6).mean() for t in thresholds]
    ax.plot(thresholds, probs, linewidth=2.5, label=scenario_name, color=color)

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Appropriation Threshold ($ Millions)', fontsize=11)
ax.set_ylabel('Probability of Exceeding', fontsize=11)
ax.set_title('(D) Exceedance Probabilities', fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'climate_scenarios_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'climate_scenarios_comparison.png'}")

# Figure 2: Disaster frequency comparison with fitted distributions
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Reload fitted distributions for this section
with open(OUTPUT_DIR / 'fitted_distributions.json', 'r') as f:
    fitted_distributions_reload = json.load(f)

# Check if we have fitted disaster frequency
disaster_freq_fit = fitted_distributions_reload.get('disaster_frequency', None)

if disaster_freq_fit is not None:
    dist_name = disaster_freq_fit['distribution']
    params = disaster_freq_fit['params']
    
    x_vals = np.arange(1, 10)  # Start from 1 (zero-truncated)
    
    # Helper function for zero-truncated Poisson PMF
    def zt_poisson_pmf(k, lam):
        return (lam**k / factorial(k)) / (1 - np.exp(-lam))
    
    # Plot smooth curves
    for scenario_name, multiplier, color in [
        ('Baseline', 1.0, 'steelblue'),
        ('Moderate (+20%)', 1.2, 'orange'),
        ('High (+40%)', 1.4, 'red')
    ]:
        if dist_name == 'zt_poisson':
            pmf = np.array([zt_poisson_pmf(k, params['lambda'] * multiplier) for k in x_vals])
            mean_disasters = params['lambda'] * multiplier / (1 - np.exp(-params['lambda'] * multiplier))
        elif dist_name == 'zt_nbinom':
            # Zero-truncated negative binomial
            n_orig = params['n']
            p_orig = params['p']
            mean_orig = n_orig * (1 - p_orig) / p_orig
            mean_new = mean_orig * multiplier
            p_new = n_orig / (n_orig + mean_new)
            # Calculate zero-truncated PMF
            p_zero = stats.nbinom.pmf(0, n_orig, p_new)
            pmf = stats.nbinom.pmf(x_vals, n_orig, p_new) / (1 - p_zero)
            mean_disasters = mean_new
        
        ax.plot(x_vals, pmf, linewidth=3, marker='o', markersize=8,
               label=f"{scenario_name} (mean={mean_disasters:.2f})",
               color=color)
    
    ax.set_xlabel('Number of Disasters per Year', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Disaster Frequency by Climate Scenario\n(Zero-Truncated {dist_name.replace("zt_", "").capitalize()} Distribution)', 
                fontweight='bold', fontsize=14)
    ax.legend(frameon=True, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xticks(x_vals)
    ax.set_xlim(0.5, 9.5)
    
    # Add note about zero-truncation
    ax.text(0.02, 0.98, 'P(X=0) = 0 (zero-truncated)\nAlways at least 1 disaster', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
else:
    # Fallback: use histograms
    for scenario_name, results in scenarios.items():
        color = {'Baseline': 'steelblue', 'Moderate (+20%)': 'orange', 'High (+40%)': 'red'}[scenario_name]
        ax.hist(results['n_disasters'], bins=range(1, 10), alpha=0.5,
               label=f"{scenario_name} (mean={results['n_disasters'].mean():.2f})",
               edgecolor='black', color=color)

    ax.set_xlabel('Number of Disasters per Year', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Disasters per Year by Climate Scenario', 
                fontweight='bold', fontsize=14)
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'climate_scenarios_disaster_frequency.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'climate_scenarios_disaster_frequency.png'}")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*80)
print("CLIMATE SCENARIO ANALYSIS COMPLETE")
print("="*80)

print(f"\nFiles saved:")
print(f"  Data:")
print(f"    - {OUTPUT_DIR / 'climate_scenario_baseline.csv'}")
print(f"    - {OUTPUT_DIR / 'climate_scenario_moderate.csv'}")
print(f"    - {OUTPUT_DIR / 'climate_scenario_high.csv'}")
print(f"    - {OUTPUT_DIR / 'climate_scenario_comparison.csv'}")
print(f"\n  Figures:")
print(f"    - {FIGURES_DIR / 'climate_scenarios_comparison.png'}")
print(f"    - {FIGURES_DIR / 'climate_scenarios_disaster_frequency.png'}")

