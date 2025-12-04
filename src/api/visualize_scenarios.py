"""
Enhanced Climate Scenario Visualizations.

Creates detailed plots showing:
1. How heatwave metrics shift across scenarios
2. Cleaner line-based appropriation comparisons
3. Distribution changes for each predictor variable
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import json
from scipy.special import factorial

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Paths
OUTPUT_DIR = Path('data/output')
FIGURES_DIR = Path('figures/climate_scenarios')
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("ENHANCED CLIMATE SCENARIO VISUALIZATIONS")
print("="*80)

# ==============================================================================
# Load results
# ==============================================================================

print("\nLoading simulation results...")

baseline_results = pd.read_csv(OUTPUT_DIR / 'climate_scenario_baseline.csv')
moderate_results = pd.read_csv(OUTPUT_DIR / 'climate_scenario_moderate.csv')
high_results = pd.read_csv(OUTPUT_DIR / 'climate_scenario_high.csv')

print(f"Loaded {len(baseline_results)} simulations per scenario")

# Load distributions for visualization
with open(OUTPUT_DIR / 'fitted_distributions.json', 'r') as f:
    fitted_distributions = json.load(f)

# ==============================================================================
# Figure 1: Heatwave Index Distribution Shifts
# ==============================================================================

print("\nCreating Figure 1: Heatwave metric distributions across scenarios...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Helper function to sample from distributions
def sample_from_fitted(fit_info, n=10000):
    """Sample from fitted distribution."""
    dist_name = fit_info['distribution']
    params = fit_info['params']
    dist = getattr(stats, dist_name)
    samples = dist.rvs(*params, size=n)
    if 'scale' in fit_info:
        samples = samples * fit_info['scale']
    return samples

# Create modified distributions for moderate and high scenarios
def modify_distribution(fit_info, multiplier, variable_type):
    """Modify distribution parameters based on climate scenario."""
    modified = fit_info.copy()
    
    if variable_type == 'peak_intensity':
        if modified['distribution'] == 'weibull_min':
            params = list(modified['params'])
            params[1] = params[1] + (multiplier - 1.0) * modified['mean']
            modified['params'] = params
        elif modified['distribution'] == 'gamma':
            params = list(modified['params'])
            params[2] = params[2] * multiplier
            modified['params'] = params
    
    elif variable_type == 'duration':
        if modified['distribution'] in ['gamma', 'lognorm']:
            params = list(modified['params'])
            params[2] = params[2] * multiplier
            modified['params'] = params
    
    elif variable_type == 'coverage':
        if modified['distribution'] == 'beta':
            params = list(modified['params'])
            params[0] = params[0] * (1 + (multiplier - 1.0))
            modified['params'] = params
    
    return modified

# Generate samples for all scenarios
n_samples = 10000

# Baseline samples
baseline_peak = sample_from_fitted(fitted_distributions['peak_intensity'], n_samples)
baseline_duration = sample_from_fitted(fitted_distributions['duration_days'], n_samples)
baseline_duration = np.clip(baseline_duration, 0, 2_600_000)
baseline_coverage = sample_from_fitted(fitted_distributions['percent_in_heatwave'], n_samples)

# Moderate samples
moderate_peak_dist = modify_distribution(fitted_distributions['peak_intensity'], 1.20, 'peak_intensity')
moderate_duration_dist = modify_distribution(fitted_distributions['duration_days'], 1.20, 'duration')
moderate_coverage_dist = modify_distribution(fitted_distributions['percent_in_heatwave'], 1.20, 'coverage')

moderate_peak = sample_from_fitted(moderate_peak_dist, n_samples)
moderate_duration = sample_from_fitted(moderate_duration_dist, n_samples)
moderate_duration = np.clip(moderate_duration, 0, 2_600_000)
moderate_coverage = sample_from_fitted(moderate_coverage_dist, n_samples)

# High samples
high_peak_dist = modify_distribution(fitted_distributions['peak_intensity'], 1.40, 'peak_intensity')
high_duration_dist = modify_distribution(fitted_distributions['duration_days'], 1.40, 'duration')
high_coverage_dist = modify_distribution(fitted_distributions['percent_in_heatwave'], 1.40, 'coverage')

high_peak = sample_from_fitted(high_peak_dist, n_samples)
high_duration = sample_from_fitted(high_duration_dist, n_samples)
high_duration = np.clip(high_duration, 0, 2_600_000)
high_coverage = sample_from_fitted(high_coverage_dist, n_samples)

# Panel A: Peak Intensity
ax = axes[0, 0]
colors = {'Baseline': 'steelblue', 'Moderate (+20%)': 'orange', 'High (+40%)': 'red'}

for label, data, color in [
    ('Baseline', baseline_peak, colors['Baseline']),
    ('Moderate (+20%)', moderate_peak, colors['Moderate (+20%)']),
    ('High (+40%)', high_peak, colors['High (+40%)'])
]:
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 500)
    ax.plot(x_range, kde(x_range), linewidth=2.5, label=label, color=color)
    ax.axvline(data.mean(), color=color, linestyle='--', alpha=0.5, linewidth=1.5)

ax.set_xlabel('Peak Intensity (°C above threshold)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(A) Peak Heatwave Intensity Distribution', fontweight='bold', fontsize=12)
ax.legend(frameon=True, loc='upper right')
ax.grid(alpha=0.3)

# Add text box with means
means_text = f"Means:\nBaseline: {baseline_peak.mean():.2f}°C\nModerate: {moderate_peak.mean():.2f}°C\nHigh: {high_peak.mean():.2f}°C"
ax.text(0.02, 0.98, means_text, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: Duration
ax = axes[0, 1]

for label, data, color in [
    ('Baseline', baseline_duration/1e6, colors['Baseline']),
    ('Moderate (+20%)', moderate_duration/1e6, colors['Moderate (+20%)']),
    ('High (+40%)', high_duration/1e6, colors['High (+40%)'])
]:
    kde = gaussian_kde(data)
    x_range = np.linspace(0, 2.5, 500)
    ax.plot(x_range, kde(x_range), linewidth=2.5, label=label, color=color)
    ax.axvline(data.mean(), color=color, linestyle='--', alpha=0.5, linewidth=1.5)

ax.set_xlabel('Duration (Million Days)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(B) Heatwave Duration Distribution', fontweight='bold', fontsize=12)
ax.legend(frameon=True, loc='upper right')
ax.grid(alpha=0.3)

means_text = f"Means:\nBaseline: {baseline_duration.mean()/1e6:.2f}M\nModerate: {moderate_duration.mean()/1e6:.2f}M\nHigh: {high_duration.mean()/1e6:.2f}M"
ax.text(0.02, 0.98, means_text, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel C: Spatial Coverage
ax = axes[1, 0]

for label, data, color in [
    ('Baseline', baseline_coverage, colors['Baseline']),
    ('Moderate (+20%)', moderate_coverage, colors['Moderate (+20%)']),
    ('High (+40%)', high_coverage, colors['High (+40%)'])
]:
    kde = gaussian_kde(data)
    x_range = np.linspace(0, 100, 500)
    ax.plot(x_range, kde(x_range), linewidth=2.5, label=label, color=color)
    ax.axvline(data.mean(), color=color, linestyle='--', alpha=0.5, linewidth=1.5)

ax.set_xlabel('Spatial Coverage (%)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(C) Heatwave Spatial Coverage Distribution', fontweight='bold', fontsize=12)
ax.legend(frameon=True, loc='upper left')
ax.grid(alpha=0.3)

means_text = f"Means:\nBaseline: {baseline_coverage.mean():.1f}%\nModerate: {moderate_coverage.mean():.1f}%\nHigh: {high_coverage.mean():.1f}%"
ax.text(0.98, 0.98, means_text, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel D: Disaster Frequency (smooth fitted curves)
ax = axes[1, 1]

# Load disaster frequency fit
disaster_freq_fit = fitted_distributions.get('disaster_frequency', None)

if disaster_freq_fit is not None:
    dist_name = disaster_freq_fit['distribution']
    params = disaster_freq_fit['params']
    
    x_vals = np.arange(1, 10)  # Start from 1 (zero-truncated)
    
    # Helper function for zero-truncated Poisson PMF
    def zt_poisson_pmf(k, lam):
        return (lam**k / factorial(k)) / (1 - np.exp(-lam))
    
    # Plot smooth PMF curves for each scenario
    for label, multiplier, color in [
        ('Baseline', 1.0, colors['Baseline']),
        ('Moderate (+20%)', 1.2, colors['Moderate (+20%)']),
        ('High (+40%)', 1.4, colors['High (+40%)'])
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
        
        ax.plot(x_vals, pmf, linewidth=3, marker='o', markersize=7,
               label=label, color=color)
    
    ax.set_xlabel('Number of Disasters per Year', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('(D) Disaster Frequency (Zero-Truncated)', fontweight='bold', fontsize=12)
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    ax.set_xticks(x_vals)
    ax.set_xlim(0.5, 9.5)
    
    # Add note about zero-truncation
    ax.text(0.02, 0.98, 'P(X=0) = 0\n(zero-truncated)', 
           transform=ax.transAxes, fontsize=8,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Add means
    means_text = f"Means:\nBaseline: {params['lambda'] if dist_name == 'poisson' else disaster_freq_fit['mean']:.2f}\n"
    means_text += f"Moderate: {params['lambda']*1.2 if dist_name == 'poisson' else disaster_freq_fit['mean']*1.2:.2f}\n"
    means_text += f"High: {params['lambda']*1.4 if dist_name == 'poisson' else disaster_freq_fit['mean']*1.4:.2f}"
    ax.text(0.98, 0.98, means_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    # Fallback: use empirical histograms
    for label, results, color in [
        ('Baseline', baseline_results, colors['Baseline']),
        ('Moderate (+20%)', moderate_results, colors['Moderate (+20%)']),
        ('High (+40%)', high_results, colors['High (+40%)'])
    ]:
        # Use histogram for discrete count data
        ax.hist(results['n_disasters'], bins=range(1, 10), alpha=0.3,
               edgecolor='black', color=color, density=True, label=None)
        
        # Overlay line
        hist, bin_edges = np.histogram(results['n_disasters'], bins=range(1, 10), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, hist, marker='o', linewidth=2.5, markersize=8,
               label=label, color=color)
        ax.axvline(results['n_disasters'].mean(), color=color, linestyle='--', 
                  alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Number of Disasters per Year', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('(D) Disaster Frequency Distribution', fontweight='bold', fontsize=12)
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)

    means_text = f"Means:\nBaseline: {baseline_results['n_disasters'].mean():.2f}\nModerate: {moderate_results['n_disasters'].mean():.2f}\nHigh: {high_results['n_disasters'].mean():.2f}"
    ax.text(0.98, 0.98, means_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'climate_heatwave_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'climate_heatwave_distributions.png'}")

# ==============================================================================
# Figure 2: Appropriation Distributions (Line Plots)
# ==============================================================================

print("\nCreating Figure 2: Appropriation distributions as smooth lines...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

scenarios = {
    'Baseline': baseline_results,
    'Moderate (+20%)': moderate_results,
    'High (+40%)': high_results
}

# Panel A: Density plot (line-based)
ax = axes[0, 0]

for scenario_name, results in scenarios.items():
    color = colors[scenario_name]
    data = results['total_appropriation'] / 1e6
    
    # Use KDE for smooth line
    kde = gaussian_kde(data)
    x_range = np.linspace(0, 500, 1000)
    density = kde(x_range)
    
    ax.plot(x_range, density, linewidth=3, label=scenario_name, color=color)
    ax.fill_between(x_range, 0, density, alpha=0.15, color=color)
    
    # Add mean line
    ax.axvline(data.mean(), color=color, linestyle='--', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Total Yearly Appropriation ($ Millions)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(A) Appropriation Distribution Comparison', fontweight='bold', fontsize=12)
ax.legend(frameon=True, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 500)

# Panel B: Cumulative distributions
ax = axes[0, 1]

for scenario_name, results in scenarios.items():
    color = colors[scenario_name]
    sorted_vals = np.sort(results['total_appropriation'] / 1e6)
    cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cumulative, linewidth=3, label=scenario_name, color=color)

# Add reference lines
for prob in [0.25, 0.5, 0.75]:
    ax.axhline(prob, color='gray', linestyle=':', alpha=0.3, linewidth=1)

ax.set_xlabel('Total Yearly Appropriation ($ Millions)', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('(B) Cumulative Distribution Functions', fontweight='bold', fontsize=12)
ax.legend(frameon=True, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 500)

# Panel C: Zoomed-in view (0-200M range)
ax = axes[1, 0]

for scenario_name, results in scenarios.items():
    color = colors[scenario_name]
    data = results['total_appropriation'] / 1e6
    
    # Use KDE for smooth line
    kde = gaussian_kde(data)
    x_range = np.linspace(0, 200, 1000)
    density = kde(x_range)
    
    ax.plot(x_range, density, linewidth=3, label=scenario_name, color=color)
    ax.fill_between(x_range, 0, density, alpha=0.15, color=color)

ax.set_xlabel('Total Yearly Appropriation ($ Millions)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(C) Distribution Detail (0-$200M)', fontweight='bold', fontsize=12)
ax.legend(frameon=True, loc='upper right')
ax.grid(alpha=0.3)

# Panel D: Exceedance probabilities
ax = axes[1, 1]

thresholds = np.arange(0, 400, 5)
for scenario_name, results in scenarios.items():
    color = colors[scenario_name]
    probs = [(results['total_appropriation'] > t*1e6).mean() for t in thresholds]
    ax.plot(thresholds, probs, linewidth=3, label=scenario_name, color=color)

# Add reference lines
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% probability')
ax.axhline(0.1, color='gray', linestyle=':', alpha=0.3, label='10% probability')

ax.set_xlabel('Appropriation Threshold ($ Millions)', fontsize=11)
ax.set_ylabel('Probability of Exceeding', fontsize=11)
ax.set_title('(D) Exceedance Probability Curves', fontweight='bold', fontsize=12)
ax.legend(frameon=True, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'climate_appropriations_lines.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'climate_appropriations_lines.png'}")

# ==============================================================================
# Figure 3: Side-by-side comparison of each metric
# ==============================================================================

print("\nCreating Figure 3: Individual metric comparisons...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Peak intensity
ax = axes[0]
for label, data, color in [
    ('Baseline', baseline_peak, colors['Baseline']),
    ('Moderate (+20%)', moderate_peak, colors['Moderate (+20%)']),
    ('High (+40%)', high_peak, colors['High (+40%)'])
]:
    kde = gaussian_kde(data)
    x_range = np.linspace(3, 12, 500)
    ax.plot(x_range, kde(x_range), linewidth=3, label=label, color=color)
    ax.fill_between(x_range, 0, kde(x_range), alpha=0.15, color=color)

ax.set_xlabel('Peak Intensity (°C above threshold)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Peak Heatwave Intensity', fontweight='bold', fontsize=13)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

# Duration
ax = axes[1]
for label, data, color in [
    ('Baseline', baseline_duration/1e6, colors['Baseline']),
    ('Moderate (+20%)', moderate_duration/1e6, colors['Moderate (+20%)']),
    ('High (+40%)', high_duration/1e6, colors['High (+40%)'])
]:
    kde = gaussian_kde(data)
    x_range = np.linspace(0, 2.5, 500)
    ax.plot(x_range, kde(x_range), linewidth=3, label=label, color=color)
    ax.fill_between(x_range, 0, kde(x_range), alpha=0.15, color=color)

ax.set_xlabel('Duration (Million Days)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Heatwave Duration', fontweight='bold', fontsize=13)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

# Spatial coverage
ax = axes[2]
for label, data, color in [
    ('Baseline', baseline_coverage, colors['Baseline']),
    ('Moderate (+20%)', moderate_coverage, colors['Moderate (+20%)']),
    ('High (+40%)', high_coverage, colors['High (+40%)'])
]:
    kde = gaussian_kde(data)
    x_range = np.linspace(20, 100, 500)
    ax.plot(x_range, kde(x_range), linewidth=3, label=label, color=color)
    ax.fill_between(x_range, 0, kde(x_range), alpha=0.15, color=color)

ax.set_xlabel('Spatial Coverage (%)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Spatial Coverage', fontweight='bold', fontsize=13)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'climate_metrics_sidebyside.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'climate_metrics_sidebyside.png'}")

# ==============================================================================
# Figure 4: Summary Statistics Table
# ==============================================================================

print("\nCreating Figure 4: Summary statistics comparison...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

# Prepare table data
table_data = []

for scenario_name, results in scenarios.items():
    mean_approp = results['total_appropriation'].mean() / 1e6
    median_approp = results['total_appropriation'].median() / 1e6
    p25 = results['total_appropriation'].quantile(0.25) / 1e6
    p75 = results['total_appropriation'].quantile(0.75) / 1e6
    mean_disasters = results['n_disasters'].mean()
    prob_50m = (results['total_appropriation'] > 50e6).mean() * 100
    prob_100m = (results['total_appropriation'] > 100e6).mean() * 100
    prob_200m = (results['total_appropriation'] > 200e6).mean() * 100
    
    table_data.append([
        scenario_name,
        f"{mean_disasters:.2f}",
        f"${mean_approp:.1f}M",
        f"${median_approp:.1f}M",
        f"${p25:.1f}M - ${p75:.1f}M",
        f"{prob_50m:.1f}%",
        f"{prob_100m:.1f}%",
        f"{prob_200m:.1f}%"
    ])

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=['Scenario', 'Avg\nDisasters/Yr', 'Mean\nApprop', 'Median\nApprop', 
              'IQR\n(25th-75th)', 'P(>$50M)', 'P(>$100M)', 'P(>$200M)'],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3)

# Style header
for i in range(8):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows by scenario
row_colors = {'Baseline': '#D6DCE4', 'Moderate (+20%)': '#FFE6CC', 'High (+40%)': '#FFCCCC'}
for i, scenario_name in enumerate(['Baseline', 'Moderate (+20%)', 'High (+40%)'], start=1):
    for j in range(8):
        table[(i, j)].set_facecolor(row_colors[scenario_name])

ax.set_title('Climate Scenario Comparison - Summary Statistics', 
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'climate_summary_table.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'climate_summary_table.png'}")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)

print(f"\nCreated 4 figures in {FIGURES_DIR}/:")
print("  1. climate_heatwave_distributions.png - 4-panel heatwave metric shifts")
print("  2. climate_appropriations_lines.png - 4-panel appropriation comparisons (line-based)")
print("  3. climate_metrics_sidebyside.png - 3-panel side-by-side metric comparison")
print("  4. climate_summary_table.png - Summary statistics table")

print("\n" + "="*80)
print("KEY VISUALIZATION FEATURES")
print("="*80)

print("\nFigure 1 (Heatwave Distributions):")
print("  ✓ Shows how peak intensity shifts right with climate change")
print("  ✓ Duration distributions become wider and shift right")
print("  ✓ Spatial coverage shifts toward higher percentages")
print("  ✓ Disaster frequency increases visibly")

print("\nFigure 2 (Appropriations - Lines):")
print("  ✓ Clean line-based density plots (much clearer than bars!)")
print("  ✓ Shaded regions show distribution spread")
print("  ✓ Zoomed panel focuses on most likely range")
print("  ✓ Exceedance curves show risk increases")

print("\nFigure 3 (Side-by-Side):")
print("  ✓ Publication-quality horizontal layout")
print("  ✓ Easy to see shifts in each metric independently")
print("  ✓ Perfect for thesis/presentation")

print("\nFigure 4 (Summary Table):")
print("  ✓ All key statistics in one place")
print("  ✓ Color-coded by scenario")
print("  ✓ Ready for thesis appendix")

print("\n" + "="*80)