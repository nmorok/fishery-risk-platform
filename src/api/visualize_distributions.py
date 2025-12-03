"""
Visualize all input distributions for Monte Carlo simulation.

This script creates comprehensive plots showing:
1. Number of disasters per year (empirical)
2. All four fitted predictor distributions
3. Historical data vs. fitted distributions
4. Sample distributions from Monte Carlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Paths
CSV_DIR = Path('data/csv')
OUTPUT_DIR = Path('data/output')
FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Load data
print("="*80)
print("VISUALIZING ALL INPUT DISTRIBUTIONS")
print("="*80)

df = pd.read_csv(CSV_DIR / 'model_data.csv')
df = df.dropna(subset=['log_appropriation', 'log_total_value', 'peak_intensity',
                       'duration_days', 'percent_in_heatwave'])
df = df[(df['peak_intensity'] > 0) & (df['log_appropriation'] > 0)]

# Load fitted distributions
with open(OUTPUT_DIR / 'fitted_distributions.json', 'r') as f:
    fitted_distributions = json.load(f)

print(f"\nUsing {len(df)} historical disasters")
print(f"Years covered: {df['sst_year'].min()} - {df['sst_year'].max()}")

# ==============================================================================
# Figure 1: Number of Disasters Per Year
# ==============================================================================

print("\nCreating Figure 1: Disasters per year distribution...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Histogram of disasters per year
ax = axes[0, 0]
disasters_per_year = df.groupby('sst_year').size()

ax.hist(disasters_per_year.values, bins=range(1, disasters_per_year.max()+2), 
        alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(disasters_per_year.mean(), color='red', linestyle='--', linewidth=2,
          label=f'Mean: {disasters_per_year.mean():.1f}')
ax.axvline(disasters_per_year.median(), color='orange', linestyle='--', linewidth=2,
          label=f'Median: {disasters_per_year.median():.0f}')

ax.set_xlabel('Number of Disasters per Year', fontsize=11)
ax.set_ylabel('Frequency (number of years)', fontsize=11)
ax.set_title('(A) Distribution of Disasters per Year', fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

# Panel B: Time series
ax = axes[0, 1]
yearly_counts = df.groupby('sst_year').size()
years = yearly_counts.index
counts = yearly_counts.values

ax.plot(years, counts, marker='o', linewidth=2, markersize=6, color='steelblue')
ax.axhline(disasters_per_year.mean(), color='red', linestyle='--', linewidth=1.5,
          alpha=0.7, label=f'Mean: {disasters_per_year.mean():.1f}')
ax.fill_between(years, disasters_per_year.mean(), alpha=0.2, color='red')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Number of Disasters', fontsize=11)
ax.set_title('(B) Disasters Over Time', fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

# Panel C: Empirical CDF
ax = axes[1, 0]
sorted_counts = np.sort(disasters_per_year.values)
cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

ax.plot(sorted_counts, cumulative, linewidth=2.5, marker='o', markersize=5,
       color='steelblue', label='Empirical CDF')

# Add reference lines
for prob in [0.25, 0.5, 0.75]:
    idx = np.searchsorted(cumulative, prob)
    if idx < len(sorted_counts):
        val = sorted_counts[idx]
        ax.axhline(prob, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(val, color='gray', linestyle=':', alpha=0.5)
        ax.scatter([val], [prob], s=100, color='red', zorder=5)

ax.set_xlabel('Number of Disasters per Year', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('(C) Empirical CDF', fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

# Panel D: Summary statistics
ax = axes[1, 1]
ax.axis('off')

stats_text = f"""
SUMMARY STATISTICS
{'='*40}

Sample size: {len(disasters_per_year)} years
Total disasters: {len(df)}

Disasters per year:
  Mean:    {disasters_per_year.mean():.2f}
  Median:  {disasters_per_year.median():.0f}
  Std Dev: {disasters_per_year.std():.2f}
  Min:     {disasters_per_year.min()}
  Max:     {disasters_per_year.max()}

Quartiles:
  25th percentile: {disasters_per_year.quantile(0.25):.0f}
  50th percentile: {disasters_per_year.quantile(0.50):.0f}
  75th percentile: {disasters_per_year.quantile(0.75):.0f}

Years with:
  1 disaster:  {(disasters_per_year == 1).sum()} years
  2 disasters: {(disasters_per_year == 2).sum()} years
  3+ disasters: {(disasters_per_year >= 3).sum()} years
"""

ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'distribution_disasters_per_year.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'distribution_disasters_per_year.png'}")

# ==============================================================================
# Figure 2: All Four Predictor Distributions
# ==============================================================================

print("\nCreating Figure 2: All predictor distributions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Helper function to plot distribution
def plot_distribution_comparison(ax, data, fit_info, var_name, xlabel):
    """Plot histogram with fitted distribution overlay."""
    
    # Histogram
    ax.hist(data, bins=30, density=True, alpha=0.6, edgecolor='black',
           color='steelblue', label='Historical data')
    
    # Fitted distribution
    dist_name = fit_info['distribution']
    params = fit_info['params']
    dist = getattr(stats, dist_name)
    
    x_range = np.linspace(data.min(), data.max(), 1000)
    pdf_fitted = dist.pdf(x_range, *params)
    
    # Apply scaling if needed
    if 'scale' in fit_info and var_name == 'percent_in_heatwave':
        pdf_fitted = pdf_fitted / fit_info['scale']  # Adjust for different x-scale
    
    ax.plot(x_range, pdf_fitted, 'r-', linewidth=2.5, 
           label=f'Fitted {dist_name}')
    
    # Add mean lines
    ax.axvline(data.mean(), color='steelblue', linestyle='--', 
              linewidth=1.5, alpha=0.7, label=f'Historical mean: {data.mean():.1f}')
    ax.axvline(fit_info['mean'], color='red', linestyle=':', 
              linewidth=1.5, alpha=0.7, label=f'Fitted mean: {fit_info["mean"]:.1f}')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{var_name.replace("_", " ").title()}', 
                fontweight='bold', fontsize=12)
    ax.legend(loc='best', frameon=True, fontsize=8)
    ax.grid(alpha=0.3)

# Panel A: log(total_value)
plot_distribution_comparison(
    axes[0, 0],
    df['log_total_value'].values,
    fitted_distributions['log_total_value'],
    'log_total_value',
    'log(Total Fishery Value)'
)

# Panel B: peak_intensity
plot_distribution_comparison(
    axes[0, 1],
    df['peak_intensity'].values,
    fitted_distributions['peak_intensity'],
    'peak_intensity',
    'Peak Heatwave Intensity (°C)'
)

# Panel C: duration_days
# Show in millions for readability
data_duration_millions = df['duration_days'].values / 1e6
ax = axes[1, 0]
ax.hist(data_duration_millions, bins=30, density=True, alpha=0.6, 
       edgecolor='black', color='steelblue', label='Historical data')

dist_name = fitted_distributions['duration_days']['distribution']
params = fitted_distributions['duration_days']['params']
dist = getattr(stats, dist_name)

x_range = np.linspace(0, data_duration_millions.max(), 1000)
pdf_fitted = dist.pdf(x_range * 1e6, *params) * 1e6  # Adjust for scaling

ax.plot(x_range, pdf_fitted, 'r-', linewidth=2.5, label=f'Fitted {dist_name}')
ax.axvline(df['duration_days'].mean() / 1e6, color='steelblue', 
          linestyle='--', linewidth=1.5, alpha=0.7,
          label=f'Historical mean: {df["duration_days"].mean()/1e6:.2f}M')
ax.axvline(fitted_distributions['duration_days']['mean'] / 1e6, 
          color='red', linestyle=':', linewidth=1.5, alpha=0.7,
          label=f'Fitted mean: {fitted_distributions["duration_days"]["mean"]/1e6:.2f}M')

ax.set_xlabel('Duration (Million Days)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Duration Days', fontweight='bold', fontsize=12)
ax.legend(loc='best', frameon=True, fontsize=8)
ax.grid(alpha=0.3)

# Panel D: percent_in_heatwave
plot_distribution_comparison(
    axes[1, 1],
    df['percent_in_heatwave'].values,
    fitted_distributions['percent_in_heatwave'],
    'percent_in_heatwave',
    'Percent Spatial Coverage (%)'
)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'distribution_all_predictors.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'distribution_all_predictors.png'}")

# ==============================================================================
# Figure 3: Sample from Distributions (what Monte Carlo will generate)
# ==============================================================================

print("\nCreating Figure 3: Monte Carlo sample distributions...")

# Generate samples from fitted distributions
n_samples = 10000

def sample_from_fitted(fit_info, n):
    """Sample from fitted distribution."""
    dist_name = fit_info['distribution']
    params = fit_info['params']
    dist = getattr(stats, dist_name)
    samples = dist.rvs(*params, size=n)
    if 'scale' in fit_info:
        samples = samples * fit_info['scale']
    return samples

# Generate samples
samples_log_value = sample_from_fitted(fitted_distributions['log_total_value'], n_samples)
samples_peak = sample_from_fitted(fitted_distributions['peak_intensity'], n_samples)
samples_duration = sample_from_fitted(fitted_distributions['duration_days'], n_samples)
samples_duration = np.clip(samples_duration, 0, 2_600_000)  # Apply safety cap
samples_percent = sample_from_fitted(fitted_distributions['percent_in_heatwave'], n_samples)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: log(total_value) samples
ax = axes[0, 0]
ax.hist(samples_log_value, bins=50, alpha=0.6, edgecolor='black', color='green',
       label=f'MC samples (n={n_samples})')
ax.hist(df['log_total_value'], bins=20, alpha=0.6, edgecolor='black', 
       color='steelblue', label='Historical data')
ax.set_xlabel('log(Total Fishery Value)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(A) log(Total Value) - MC Samples vs Historical', 
            fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

# Panel B: peak_intensity samples
ax = axes[0, 1]
ax.hist(samples_peak, bins=50, alpha=0.6, edgecolor='black', color='green',
       label=f'MC samples (n={n_samples})')
ax.hist(df['peak_intensity'], bins=20, alpha=0.6, edgecolor='black',
       color='steelblue', label='Historical data')
ax.set_xlabel('Peak Intensity (°C)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(B) Peak Intensity - MC Samples vs Historical',
            fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

# Panel C: duration samples (in millions)
ax = axes[1, 0]
ax.hist(samples_duration / 1e6, bins=50, alpha=0.6, edgecolor='black', color='green',
       label=f'MC samples (n={n_samples})')
ax.hist(df['duration_days'] / 1e6, bins=20, alpha=0.6, edgecolor='black',
       color='steelblue', label='Historical data')
ax.set_xlabel('Duration (Million Days)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(C) Duration - MC Samples vs Historical',
            fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

# Panel D: percent samples
ax = axes[1, 1]
ax.hist(samples_percent, bins=50, alpha=0.6, edgecolor='black', color='green',
       label=f'MC samples (n={n_samples})')
ax.hist(df['percent_in_heatwave'], bins=20, alpha=0.6, edgecolor='black',
       color='steelblue', label='Historical data')
ax.set_xlabel('Percent in Heatwave (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(D) Spatial Coverage - MC Samples vs Historical',
            fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'distribution_monte_carlo_samples.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'distribution_monte_carlo_samples.png'}")

# ==============================================================================
# Figure 4: Distribution Statistics Summary Table
# ==============================================================================

print("\nCreating Figure 4: Summary statistics table...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Prepare data for table
table_data = []

# Disasters per year
table_data.append([
    'Disasters/Year',
    'Empirical',
    '-',
    f"{disasters_per_year.mean():.2f}",
    f"{disasters_per_year.std():.2f}",
    f"{disasters_per_year.min()}-{disasters_per_year.max()}",
    f"{disasters_per_year.quantile(0.05):.0f}, {disasters_per_year.quantile(0.95):.0f}"
])

# log(total_value)
table_data.append([
    'log(Total Value)',
    fitted_distributions['log_total_value']['distribution'].upper(),
    f"μ={fitted_distributions['log_total_value']['params'][0]:.2f}, σ={fitted_distributions['log_total_value']['params'][1]:.2f}",
    f"{df['log_total_value'].mean():.2f}",
    f"{df['log_total_value'].std():.2f}",
    f"{df['log_total_value'].min():.1f}-{df['log_total_value'].max():.1f}",
    f"{df['log_total_value'].quantile(0.05):.1f}, {df['log_total_value'].quantile(0.95):.1f}"
])

# peak_intensity
dist_name = fitted_distributions['peak_intensity']['distribution']
params_str = ', '.join([f"{p:.2f}" for p in fitted_distributions['peak_intensity']['params'][:2]])
table_data.append([
    'Peak Intensity (°C)',
    dist_name.upper(),
    params_str,
    f"{df['peak_intensity'].mean():.2f}",
    f"{df['peak_intensity'].std():.2f}",
    f"{df['peak_intensity'].min():.1f}-{df['peak_intensity'].max():.1f}",
    f"{df['peak_intensity'].quantile(0.05):.1f}, {df['peak_intensity'].quantile(0.95):.1f}"
])

# duration_days
dist_name = fitted_distributions['duration_days']['distribution']
table_data.append([
    'Duration (days)',
    dist_name.upper(),
    'See params above',
    f"{df['duration_days'].mean()/1e6:.2f}M",
    f"{df['duration_days'].std()/1e6:.2f}M",
    f"{df['duration_days'].min()/1e3:.0f}K-{df['duration_days'].max()/1e6:.1f}M",
    f"{df['duration_days'].quantile(0.05)/1e3:.0f}K, {df['duration_days'].quantile(0.95)/1e6:.1f}M"
])

# percent_in_heatwave
table_data.append([
    'Coverage (%)',
    fitted_distributions['percent_in_heatwave']['distribution'].upper(),
    'Beta params',
    f"{df['percent_in_heatwave'].mean():.1f}%",
    f"{df['percent_in_heatwave'].std():.1f}%",
    f"{df['percent_in_heatwave'].min():.1f}-{df['percent_in_heatwave'].max():.1f}",
    f"{df['percent_in_heatwave'].quantile(0.05):.0f}, {df['percent_in_heatwave'].quantile(0.95):.0f}"
])

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=['Variable', 'Distribution', 'Parameters', 'Mean', 'Std Dev', 
              'Range', '5th, 95th %ile'],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    color = '#E7E6E6' if i % 2 == 0 else 'white'
    for j in range(7):
        table[(i, j)].set_facecolor(color)

ax.set_title('Distribution Summary Statistics', 
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'distribution_summary_table.png', dpi=300, bbox_inches='tight')
print(f"Saved to {FIGURES_DIR / 'distribution_summary_table.png'}")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)

print(f"\nCreated 4 figures in {FIGURES_DIR}/:")
print("  1. distribution_disasters_per_year.png")
print("  2. distribution_all_predictors.png")
print("  3. distribution_monte_carlo_samples.png")
print("  4. distribution_summary_table.png")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print(f"\nDisasters per year:")
print(f"  - Average: {disasters_per_year.mean():.1f} disasters/year")
print(f"  - Range: {disasters_per_year.min()} to {disasters_per_year.max()}")
print(f"  - Most common: {disasters_per_year.mode()[0]} disasters/year")

print(f"\nPredictor distributions:")
print(f"  - log(value): {fitted_distributions['log_total_value']['distribution']} "
      f"(mean={fitted_distributions['log_total_value']['mean']:.1f})")
print(f"  - peak_intensity: {fitted_distributions['peak_intensity']['distribution']} "
      f"(mean={fitted_distributions['peak_intensity']['mean']:.1f}°C)")
print(f"  - duration: {fitted_distributions['duration_days']['distribution']} "
      f"(mean={fitted_distributions['duration_days']['mean']/1e6:.2f}M days)")
print(f"  - coverage: {fitted_distributions['percent_in_heatwave']['distribution']} "
      f"(mean={fitted_distributions['percent_in_heatwave']['mean']:.1f}%)")

print("\n" + "="*80)