"""
Fit parametric distributions to historical disaster characteristics.

This script:
1. Analyzes the distribution of each predictor variable
2. Fits appropriate parametric distributions
3. Tests goodness-of-fit
4. Saves parameters for Monte Carlo sampling
5. Creates visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, anderson
import json

# Paths
CSV_DIR = Path('data/csv')
OUTPUT_DIR = Path('data/output')
FIGURES_DIR = Path('figures/distributions')
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Load data
print("="*80)
print("FITTING DISTRIBUTIONS TO HISTORICAL DATA")
print("="*80)

df = pd.read_csv(CSV_DIR / 'model_data.csv')
df = df.dropna(subset=['log_appropriation', 'log_total_value', 'peak_intensity',
                       'duration_days', 'percent_in_heatwave'])
df = df[(df['peak_intensity'] > 0) & (df['log_appropriation'] > 0)]

print(f"\nUsing {len(df)} historical disasters")

# ==============================================================================
# Helper Functions
# ==============================================================================

def fit_and_evaluate_distributions(data, var_name, distributions_to_try):
    """
    Fit multiple distributions and return the best one.
    
    Parameters:
    -----------
    data : array
        The data to fit
    var_name : str
        Name of variable (for printing)
    distributions_to_try : list
        List of scipy.stats distribution objects to try
        
    Returns:
    --------
    dict : Best distribution with parameters and fit statistics
    """
    
    best_dist = None
    best_ks_stat = np.inf
    best_params = None
    
    results = []
    
    for dist in distributions_to_try:
        try:
            # Fit distribution
            params = dist.fit(data)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = kstest(data, dist.name, args=params)
            
            # Anderson-Darling test
            ad_result = anderson(data, dist=dist.name if dist.name in ['norm', 'expon'] else 'norm')
            
            # Log-likelihood
            log_likelihood = np.sum(dist.logpdf(data, *params))
            
            # AIC and BIC
            k = len(params)  # number of parameters
            n = len(data)
            aic = 2*k - 2*log_likelihood
            bic = k*np.log(n) - 2*log_likelihood
            
            results.append({
                'distribution': dist.name,
                'params': params,
                'ks_stat': ks_stat,
                'ks_pval': ks_pval,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic
            })
            
            if ks_stat < best_ks_stat:
                best_ks_stat = ks_stat
                best_dist = dist.name
                best_params = params
                
        except Exception as e:
            print(f"  Warning: Could not fit {dist.name} - {e}")
            continue
    
    # Sort by AIC (lower is better)
    results.sort(key=lambda x: x['aic'])
    
    return results, best_dist, best_params


def visualize_fit(data, dist_name, params, var_name, filename):
    """
    Create visualization of fitted distribution vs. data.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get distribution object
    dist = getattr(stats, dist_name)
    
    # Panel 1: Histogram with fitted PDF
    ax = axes[0, 0]
    ax.hist(data, bins=30, density=True, alpha=0.7, edgecolor='black', 
            color='steelblue', label='Historical data')
    
    x_range = np.linspace(data.min(), data.max(), 1000)
    pdf_fitted = dist.pdf(x_range, *params)
    ax.plot(x_range, pdf_fitted, 'r-', linewidth=2, label=f'Fitted {dist_name}')
    
    ax.set_xlabel(var_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'(A) Histogram with Fitted {dist_name.title()}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: Q-Q plot
    ax = axes[0, 1]
    
    # Generate theoretical quantiles
    sorted_data = np.sort(data)
    theoretical_quantiles = dist.ppf(np.linspace(0.01, 0.99, len(data)), *params)
    
    ax.scatter(theoretical_quantiles, sorted_data, alpha=0.6, s=30, color='steelblue')
    
    # Add 1:1 line
    min_val = min(theoretical_quantiles.min(), sorted_data.min())
    max_val = max(theoretical_quantiles.max(), sorted_data.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Sample Quantiles', fontsize=12)
    ax.set_title('(B) Q-Q Plot', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: CDF comparison
    ax = axes[1, 0]
    
    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    ax.plot(sorted_data, empirical_cdf, linewidth=2, label='Empirical CDF', color='steelblue')
    
    x_range = np.linspace(data.min(), data.max(), 1000)
    fitted_cdf = dist.cdf(x_range, *params)
    ax.plot(x_range, fitted_cdf, 'r--', linewidth=2, label=f'Fitted {dist_name} CDF')
    
    ax.set_xlabel(var_name, fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('(C) CDF Comparison', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 4: Residuals
    ax = axes[1, 1]
    
    # Calculate residuals (empirical - theoretical)
    sorted_data = np.sort(data)
    theoretical_quantiles = dist.ppf(np.linspace(0.01, 0.99, len(data)), *params)
    residuals = sorted_data - theoretical_quantiles
    
    ax.scatter(theoretical_quantiles, residuals, alpha=0.6, s=30, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('(D) Residual Plot', fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to {filename}")


# ==============================================================================
# Fit distributions for each variable
# ==============================================================================

fitted_distributions = {}

print("\n" + "="*80)
print("1. LOG(TOTAL_VALUE)")
print("="*80)

data = df['log_total_value'].values
print(f"\nDescriptive statistics:")
print(f"  Mean: {data.mean():.3f}")
print(f"  Std: {data.std():.3f}")
print(f"  Median: {np.median(data):.3f}")
print(f"  Min: {data.min():.3f}")
print(f"  Max: {data.max():.3f}")

# Try normal, lognormal (of original values)
distributions = [stats.norm]
results, best_dist, best_params = fit_and_evaluate_distributions(
    data, 'log(total_value)', distributions
)

print(f"\nBest fit: {best_dist}")
print(f"Parameters: {best_params}")
print(f"  Mean: {best_params[0]:.3f}")
print(f"  Std: {best_params[1]:.3f}")

fitted_distributions['log_total_value'] = {
    'distribution': best_dist,
    'params': best_params,
    'mean': best_params[0],
    'std': best_params[1]
}

visualize_fit(data, best_dist, best_params, 'log(total_value)', 
              FIGURES_DIR / 'fit_log_total_value.png')

# ==============================================================================
print("\n" + "="*80)
print("2. PEAK_INTENSITY")
print("="*80)

data = df['peak_intensity'].values
print(f"\nDescriptive statistics:")
print(f"  Mean: {data.mean():.3f}")
print(f"  Std: {data.std():.3f}")
print(f"  Median: {np.median(data):.3f}")
print(f"  Min: {data.min():.3f}")
print(f"  Max: {data.max():.3f}")
print(f"  Skewness: {stats.skew(data):.3f}")

# Try gamma, lognormal, exponential
distributions = [stats.gamma, stats.lognorm, stats.weibull_min]
results, best_dist, best_params = fit_and_evaluate_distributions(
    data, 'peak_intensity', distributions
)

print(f"\nFitting results (by AIC):")
for i, r in enumerate(results[:3]):
    print(f"  {i+1}. {r['distribution']}: AIC={r['aic']:.2f}, KS={r['ks_stat']:.4f} (p={r['ks_pval']:.4f})")

print(f"\nBest fit: {best_dist}")
print(f"Parameters: {best_params}")

fitted_distributions['peak_intensity'] = {
    'distribution': best_dist,
    'params': best_params,
}

# Add convenient summary stats
dist = getattr(stats, best_dist)
mean_val = dist.mean(*best_params)
std_val = dist.std(*best_params)
print(f"  Implied mean: {mean_val:.3f}")
print(f"  Implied std: {std_val:.3f}")
fitted_distributions['peak_intensity']['mean'] = mean_val
fitted_distributions['peak_intensity']['std'] = std_val

visualize_fit(data, best_dist, best_params, 'peak_intensity',
              FIGURES_DIR / 'fit_peak_intensity.png')

# ==============================================================================
print("\n" + "="*80)
print("3. DURATION_DAYS")
print("="*80)

data = df['duration_days'].values
print(f"\nDescriptive statistics:")
print(f"  Mean: {data.mean():.1f}")
print(f"  Std: {data.std():.1f}")
print(f"  Median: {np.median(data):.1f}")
print(f"  Min: {data.min():.1f}")
print(f"  Max: {data.max():.1f}")
print(f"  Skewness: {stats.skew(data):.3f}")

# Try gamma, lognormal
distributions = [stats.gamma, stats.lognorm, stats.expon]
results, best_dist, best_params = fit_and_evaluate_distributions(
    data, 'duration_days', distributions
)

print(f"\nFitting results (by AIC):")
for i, r in enumerate(results[:3]):
    print(f"  {i+1}. {r['distribution']}: AIC={r['aic']:.2f}, KS={r['ks_stat']:.4f} (p={r['ks_pval']:.4f})")

# Force gamma to avoid extreme lognormal values
print(f"\nForcing gamma distribution to prevent extreme outliers")
best_dist = 'gamma'
best_params = stats.gamma.fit(data, floc=0)
print(f"Parameters: {best_params}")

fitted_distributions['duration_days'] = {
    'distribution': best_dist,
    'params': best_params,
}

dist = getattr(stats, best_dist)
mean_val = dist.mean(*best_params)
std_val = dist.std(*best_params)
print(f"  Implied mean: {mean_val:.1f}")
print(f"  Implied std: {std_val:.1f}")
fitted_distributions['duration_days']['mean'] = mean_val
fitted_distributions['duration_days']['std'] = std_val

visualize_fit(data, best_dist, best_params, 'duration_days',
              FIGURES_DIR / 'fit_duration_days.png')

# ==============================================================================
print("\n" + "="*80)
print("4. PERCENT_IN_HEATWAVE")
print("="*80)

data = df['percent_in_heatwave'].values
print(f"\nDescriptive statistics:")
print(f"  Mean: {data.mean():.2f}%")
print(f"  Std: {data.std():.2f}%")
print(f"  Median: {np.median(data):.2f}%")
print(f"  Min: {data.min():.2f}%")
print(f"  Max: {data.max():.2f}%")

# Try beta (rescaled to 0-1), truncated normal, or normal
# For beta, need to scale to (0,1)
data_scaled = data / 100.0

# For highly skewed data near 1, try normal directly on original scale
# This often works better than beta for data concentrated near boundary
distributions = [stats.norm, stats.lognorm]

# Try fitting on original scale first
results_original, best_dist_orig, best_params_orig = fit_and_evaluate_distributions(
    data, 'percent_in_heatwave (original scale)', distributions
)

# Also try on scaled data with beta
distributions_scaled = [stats.beta]
results_scaled, best_dist_scaled, best_params_scaled = fit_and_evaluate_distributions(
    data_scaled, 'percent_in_heatwave (scaled)', distributions_scaled
)

# Compare AICs and choose best
if results_original[0]['aic'] < results_scaled[0]['aic']:
    print(f"\nUsing distribution on original scale (AIC={results_original[0]['aic']:.2f})")
    best_dist = best_dist_orig
    best_params = best_params_orig
    use_scaling = False
else:
    print(f"\nUsing beta on scaled data (AIC={results_scaled[0]['aic']:.2f})")
    best_dist = best_dist_scaled
    best_params = best_params_scaled
    use_scaling = True

print(f"\nFitting results:")
if not use_scaling:
    for i, r in enumerate(results_original[:2]):
        print(f"  {i+1}. {r['distribution']}: AIC={r['aic']:.2f}, KS={r['ks_stat']:.4f} (p={r['ks_pval']:.4f})")
else:
    for i, r in enumerate(results_scaled[:2]):
        print(f"  {i+1}. {r['distribution']}: AIC={r['aic']:.2f}, KS={r['ks_stat']:.4f} (p={r['ks_pval']:.4f})")

print(f"\nBest fit: {best_dist}")
print(f"Parameters: {best_params}")

fitted_distributions['percent_in_heatwave'] = {
    'distribution': best_dist,
    'params': best_params,
}

if use_scaling:
    fitted_distributions['percent_in_heatwave']['scale'] = 100.0  # Need to multiply samples by 100
    print("  (Note: Samples will be scaled by 100)")

# Visualize on original scale
visualize_fit(data, best_dist, [p for p in best_params], 'percent_in_heatwave',
              FIGURES_DIR / 'fit_percent_in_heatwave.png')

dist = getattr(stats, best_dist)
if best_dist == 'beta' and use_scaling:
    mean_val = dist.mean(*best_params) * 100
    std_val = dist.std(*best_params) * 100
elif best_dist == 'norm':
    mean_val = best_params[0]  # Already on correct scale
    std_val = best_params[1]
else:
    mean_val = dist.mean(*best_params)
    std_val = dist.std(*best_params)
print(f"  Implied mean: {mean_val:.2f}%")
print(f"  Implied std: {std_val:.2f}%")
fitted_distributions['percent_in_heatwave']['mean'] = mean_val
fitted_distributions['percent_in_heatwave']['std'] = std_val

# ==============================================================================
# Save fitted distributions
# ==============================================================================

print("\n" + "="*80)
print("SAVING FITTED DISTRIBUTIONS")
print("="*80)

# Convert numpy arrays to lists for JSON serialization
distributions_json = {}
for var, fit_info in fitted_distributions.items():
    distributions_json[var] = {
        'distribution': fit_info['distribution'],
        'params': [float(p) for p in fit_info['params']],
        'mean': float(fit_info['mean']),
        'std': float(fit_info['std'])
    }
    if 'scale' in fit_info:
        distributions_json[var]['scale'] = float(fit_info['scale'])

with open(OUTPUT_DIR / 'fitted_distributions.json', 'w') as f:
    json.dump(distributions_json, f, indent=2)

print(f"\nSaved fitted distributions to {OUTPUT_DIR / 'fitted_distributions.json'}")

# Also save as a readable text file
with open(OUTPUT_DIR / 'fitted_distributions.txt', 'w') as f:
    f.write("FITTED DISTRIBUTIONS FOR MONTE CARLO SIMULATION\n")
    f.write("="*80 + "\n\n")
    
    for var, fit_info in fitted_distributions.items():
        f.write(f"{var}:\n")
        f.write(f"  Distribution: {fit_info['distribution']}\n")
        f.write(f"  Parameters: {fit_info['params']}\n")
        f.write(f"  Mean: {fit_info['mean']:.3f}\n")
        f.write(f"  Std: {fit_info['std']:.3f}\n")
        if 'scale' in fit_info:
            f.write(f"  Scale: {fit_info['scale']}\n")
        f.write("\n")

print(f"Saved readable summary to {OUTPUT_DIR / 'fitted_distributions.txt'}")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nFitted distributions:")
for var, fit_info in fitted_distributions.items():
    print(f"\n{var}:")
    print(f"  → {fit_info['distribution'].upper()} distribution")
    print(f"  → Mean: {fit_info['mean']:.3f}, Std: {fit_info['std']:.3f}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Review fitted distributions in figures/distributions/")
print("2. Check goodness-of-fit (Q-Q plots, residuals)")
print("3. Use 'fitted_distributions.json' in Monte Carlo simulation")
print("4. Run: python stan_posterior_utils_with_distributions.py")
print("\n" + "="*80)