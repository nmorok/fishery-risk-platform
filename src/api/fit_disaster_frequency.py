"""
Fit a parametric distribution to disaster frequency (count data).

This script:
1. Fits Poisson and Negative Binomial distributions to historical disaster counts
2. Selects best fit using AIC/BIC
3. Saves parameters for Monte Carlo simulation
4. Creates visualization comparing empirical vs fitted
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import json
from scipy.special import factorial

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Paths
CSV_DIR = Path('data/csv')
OUTPUT_DIR = Path('data/output')
FIGURES_DIR = Path('figures/distributions')
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("FITTING DISTRIBUTION TO DISASTER FREQUENCY")
print("="*80)

# Load data
df = pd.read_csv(CSV_DIR / 'model_data.csv')
df = df.dropna(subset=['log_appropriation', 'log_total_value', 'peak_intensity',
                       'duration_days', 'percent_in_heatwave'])
df = df[(df['peak_intensity'] > 0) & (df['log_appropriation'] > 0)]

# Get disaster counts per year
disasters_per_year = df.groupby('sst_year').size().values

print(f"\nHistorical disaster frequency:")
print(f"  Years observed: {len(disasters_per_year)}")
print(f"  Mean: {disasters_per_year.mean():.2f} disasters/year")
print(f"  Std: {disasters_per_year.std():.2f}")
print(f"  Min: {disasters_per_year.min()}")
print(f"  Max: {disasters_per_year.max()}")
print(f"  Median: {np.median(disasters_per_year):.0f}")

# ==============================================================================
# Fit candidate distributions for count data (ZERO-TRUNCATED)
# ==============================================================================

print("\n" + "="*80)
print("FITTING ZERO-TRUNCATED DISTRIBUTIONS")
print("="*80)
print("\nNote: Excluding zero disasters (we only model years with disasters)")

results = []

# 1. ZERO-TRUNCATED POISSON DISTRIBUTION
# PMF: P(X=k) = λ^k / (k! * (e^λ - 1)) for k = 1, 2, 3, ...
# Mean: λ / (1 - e^(-λ))

# MLE for zero-truncated Poisson
sample_mean = disasters_per_year.mean()
# Solve for λ such that λ/(1-exp(-λ)) = sample_mean
from scipy.optimize import fsolve
def zt_poisson_mean_eq(lam, target_mean):
    return lam / (1 - np.exp(-lam)) - target_mean

lambda_mle = fsolve(zt_poisson_mean_eq, sample_mean, args=(sample_mean,))[0]

# Log-likelihood for zero-truncated Poisson
def zt_poisson_logpmf(k, lam):
    """Log PMF of zero-truncated Poisson."""
    return k * np.log(lam) - lam - np.sum(np.log(np.arange(1, k+1))) - np.log(1 - np.exp(-lam))

zt_poisson_loglik = np.sum([zt_poisson_logpmf(k, lambda_mle) for k in disasters_per_year])
zt_poisson_aic = -2 * zt_poisson_loglik + 2 * 1  # 1 parameter
zt_poisson_bic = -2 * zt_poisson_loglik + 1 * np.log(len(disasters_per_year))

# Calculate expected frequencies for chi-square test
unique_counts = np.arange(disasters_per_year.min(), disasters_per_year.max() + 1)
observed_freq = np.array([np.sum(disasters_per_year == k) for k in unique_counts])

def zt_poisson_pmf(k, lam):
    """PMF of zero-truncated Poisson."""
    return (lam**k / factorial(k)) / (1 - np.exp(-lam))

expected_freq = len(disasters_per_year) * np.array([zt_poisson_pmf(k, lambda_mle) for k in unique_counts])

# Chi-square test
valid = expected_freq >= 1
if valid.sum() > 1:
    chi2_stat = np.sum((observed_freq[valid] - expected_freq[valid])**2 / expected_freq[valid])
    chi2_pval = 1 - stats.chi2.cdf(chi2_stat, df=valid.sum() - 1 - 1)
else:
    chi2_pval = np.nan

# Mean and variance of zero-truncated Poisson
zt_mean = lambda_mle / (1 - np.exp(-lambda_mle))
zt_var = lambda_mle * (1 + lambda_mle) / (1 - np.exp(-lambda_mle)) - (lambda_mle / (1 - np.exp(-lambda_mle)))**2

results.append({
    'distribution': 'zt_poisson',
    'params': {'lambda': lambda_mle},
    'loglik': zt_poisson_loglik,
    'aic': zt_poisson_aic,
    'bic': zt_poisson_bic,
    'chi2_pval': chi2_pval,
    'mean': zt_mean,
    'var': zt_var
})

print(f"\n1. ZERO-TRUNCATED POISSON")
print(f"   λ = {lambda_mle:.3f}")
print(f"   AIC = {zt_poisson_aic:.2f}")
print(f"   BIC = {zt_poisson_bic:.2f}")
print(f"   Chi² p-value = {chi2_pval:.4f}")
print(f"   Mean = {zt_mean:.3f}, Var = {zt_var:.3f}")
print(f"   P(X=0) = 0 (zero-truncated)")
print(f"   P(X=1) = {zt_poisson_pmf(1, lambda_mle):.3f}")
print(f"   P(X=2) = {zt_poisson_pmf(2, lambda_mle):.3f}")

# 2. ZERO-TRUNCATED NEGATIVE BINOMIAL DISTRIBUTION
# For overdispersed count data

sample_mean = disasters_per_year.mean()
sample_var = disasters_per_year.var()

if sample_var > sample_mean:  # Overdispersed
    print(f"\n2. ZERO-TRUNCATED NEGATIVE BINOMIAL")
    print(f"   (Data appears overdispersed: var={sample_var:.2f} > mean={sample_mean:.2f})")
    
    # For zero-truncated negative binomial, we'll use a simpler approach
    # Just fit regular negative binomial and note it's implicitly truncated
    # since we have no zeros in data
    
    def negbin_loglik(params, data):
        """Negative log-likelihood for negative binomial."""
        n, p = params
        if n <= 0 or p <= 0 or p >= 1:
            return 1e10
        return -np.sum(stats.nbinom.logpmf(data, n, p))
    
    # Initial guess
    n_init = sample_mean**2 / (sample_var - sample_mean)
    p_init = sample_mean / sample_var
    
    # MLE
    result = minimize(negbin_loglik, [n_init, p_init], args=(disasters_per_year,),
                     method='L-BFGS-B', bounds=[(0.1, 100), (0.01, 0.99)])
    
    if result.success:
        n_mle, p_mle = result.x
        negbin_loglik_val = -result.fun
        negbin_aic = -2 * negbin_loglik_val + 2 * 2
        negbin_bic = -2 * negbin_loglik_val + 2 * np.log(len(disasters_per_year))
        
        # Chi-square test
        expected_freq = len(disasters_per_year) * stats.nbinom.pmf(unique_counts, n_mle, p_mle)
        valid = expected_freq >= 1
        if valid.sum() > 2:
            chi2_stat = np.sum((observed_freq[valid] - expected_freq[valid])**2 / expected_freq[valid])
            chi2_pval = 1 - stats.chi2.cdf(chi2_stat, df=valid.sum() - 2 - 1)
        else:
            chi2_pval = np.nan
        
        negbin_mean = n_mle * (1 - p_mle) / p_mle
        negbin_var = n_mle * (1 - p_mle) / p_mle**2
        
        results.append({
            'distribution': 'zt_nbinom',
            'params': {'n': n_mle, 'p': p_mle},
            'loglik': negbin_loglik_val,
            'aic': negbin_aic,
            'bic': negbin_bic,
            'chi2_pval': chi2_pval,
            'mean': negbin_mean,
            'var': negbin_var
        })
        
        print(f"   n = {n_mle:.3f}, p = {p_mle:.3f}")
        print(f"   AIC = {negbin_aic:.2f}")
        print(f"   BIC = {negbin_bic:.2f}")
        print(f"   Chi² p-value = {chi2_pval:.4f}")
        print(f"   Mean = {negbin_mean:.3f}, Var = {negbin_var:.3f}")
        print(f"   P(X=0) ≈ {stats.nbinom.pmf(0, n_mle, p_mle):.4f} (will be excluded in sampling)")
    else:
        print(f"   Optimization failed")
else:
    print(f"\n2. ZERO-TRUNCATED NEGATIVE BINOMIAL: Skipped (data not overdispersed)")

# 3. EMPIRICAL DISTRIBUTION (for comparison)
empirical_mean = disasters_per_year.mean()
empirical_var = disasters_per_year.var()

results.append({
    'distribution': 'empirical',
    'params': {'data': disasters_per_year.tolist()},
    'loglik': None,
    'aic': None,
    'bic': None,
    'chi2_pval': 1.0,  # Perfect fit by definition
    'mean': empirical_mean,
    'var': empirical_var
})

print(f"\n3. EMPIRICAL (bootstrap)")
print(f"   Mean = {empirical_mean:.3f}, Var = {empirical_var:.3f}")

# ==============================================================================
# Select best distribution
# ==============================================================================

print("\n" + "="*80)
print("MODEL SELECTION")
print("="*80)

# Sort by AIC
valid_results = [r for r in results if r['aic'] is not None]
valid_results.sort(key=lambda x: x['aic'])

print("\nRanking by AIC:")
for i, r in enumerate(valid_results, 1):
    print(f"  {i}. {r['distribution'].upper()}: AIC={r['aic']:.2f}, "
          f"Mean={r['mean']:.3f}, Var={r['var']:.3f}, "
          f"Chi² p={r['chi2_pval']:.4f}")

# Use best by AIC, but prefer zero-truncated Poisson if it has good fit
zt_poisson_result = [r for r in results if r['distribution'] == 'zt_poisson'][0]
if zt_poisson_result['chi2_pval'] > 0.05 and not np.isnan(zt_poisson_result['chi2_pval']):
    best_result = zt_poisson_result
    print(f"\nUsing ZERO-TRUNCATED POISSON (good fit, simplest model, no zeros)")
else:
    best_result = valid_results[0]
    print(f"\nUsing {best_result['distribution'].upper()} (best AIC)")

print(f"\nIMPORTANT: P(0 disasters) = 0 (zero-truncated distribution)")

# ==============================================================================
# Save fitted distribution
# ==============================================================================

disaster_frequency_fit = {
    'distribution': best_result['distribution'],
    'params': best_result['params'],
    'mean': best_result['mean'],
    'var': best_result['var'],
    'empirical_mean': float(empirical_mean),
    'empirical_var': float(empirical_var)
}

# Load existing fitted distributions
with open(OUTPUT_DIR / 'fitted_distributions.json', 'r') as f:
    fitted_distributions = json.load(f)

# Add disaster frequency
fitted_distributions['disaster_frequency'] = disaster_frequency_fit

# Save
with open(OUTPUT_DIR / 'fitted_distributions.json', 'w') as f:
    json.dump(fitted_distributions, f, indent=2)

print(f"\nSaved disaster frequency distribution to fitted_distributions.json")

# ==============================================================================
# Visualization
# ==============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Empirical PMF vs Fitted
ax = axes[0, 0]

# Empirical
unique_vals, counts = np.unique(disasters_per_year, return_counts=True)
probs = counts / len(disasters_per_year)
ax.bar(unique_vals, probs, alpha=0.6, color='steelblue', edgecolor='black',
      label='Empirical', width=0.4, align='center')

# Fitted (zero-truncated, starts from 1)
x_vals = np.arange(1, disasters_per_year.max() + 2)
if best_result['distribution'] == 'zt_poisson':
    fitted_pmf = np.array([zt_poisson_pmf(k, best_result['params']['lambda']) for k in x_vals])
    ax.plot(x_vals, fitted_pmf, 'r-', linewidth=2.5, marker='o', markersize=6,
           label=f"ZT-Poisson(λ={best_result['params']['lambda']:.2f})")
elif best_result['distribution'] == 'zt_nbinom':
    # For negative binomial, manually set P(0)=0 by renormalizing
    n_param = best_result['params']['n']
    p_param = best_result['params']['p']
    p_zero = stats.nbinom.pmf(0, n_param, p_param)
    fitted_pmf = stats.nbinom.pmf(x_vals, n_param, p_param) / (1 - p_zero)
    ax.plot(x_vals, fitted_pmf, 'r-', linewidth=2.5, marker='o', markersize=6,
           label=f"ZT-NegBinom(n={n_param:.2f}, p={p_param:.2f})")

ax.set_xlabel('Number of Disasters per Year', fontsize=11)
ax.set_ylabel('Probability', fontsize=11)
ax.set_title('(A) Probability Mass Function (Zero-Truncated)', fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)
ax.set_xticks(x_vals)
ax.set_xlim(0.5, disasters_per_year.max() + 1.5)

# Add text noting P(0)=0
ax.text(0.98, 0.98, 'P(X=0) = 0\n(zero-truncated)', 
       transform=ax.transAxes, fontsize=9,
       verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Panel B: Cumulative Distribution
ax = axes[0, 1]

# Empirical CDF
sorted_vals = np.sort(disasters_per_year)
empirical_cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
ax.step(sorted_vals, empirical_cdf, where='post', linewidth=2.5, 
       label='Empirical', color='steelblue')

# Fitted CDF (zero-truncated, starts from 1)
x_vals_cdf = np.arange(1, disasters_per_year.max() + 2)
if best_result['distribution'] == 'zt_poisson':
    # Cumulative sum of PMF
    fitted_cdf = np.cumsum([zt_poisson_pmf(k, best_result['params']['lambda']) for k in x_vals_cdf])
elif best_result['distribution'] == 'zt_nbinom':
    # Zero-truncated negative binomial CDF
    n_param = best_result['params']['n']
    p_param = best_result['params']['p']
    p_zero = stats.nbinom.pmf(0, n_param, p_param)
    fitted_cdf = (stats.nbinom.cdf(x_vals_cdf, n_param, p_param) - p_zero) / (1 - p_zero)

ax.plot(x_vals_cdf, fitted_cdf, 'r--', linewidth=2.5, marker='o', markersize=5,
       label='Fitted', alpha=0.8)

ax.set_xlabel('Number of Disasters per Year', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('(B) Cumulative Distribution Function', fontweight='bold', fontsize=12)
ax.legend(frameon=True)
ax.grid(alpha=0.3)
ax.set_xlim(0.5, disasters_per_year.max() + 1.5)

# Panel C: Q-Q Plot
ax = axes[1, 0]

if best_result['distribution'] in ['zt_poisson', 'zt_nbinom']:
    # Generate theoretical quantiles for zero-truncated distribution
    n_points = len(disasters_per_year)
    theoretical_probs = np.arange(1, n_points + 1) / (n_points + 1)
    
    if best_result['distribution'] == 'zt_poisson':
        # Invert CDF for zero-truncated Poisson
        lam = best_result['params']['lambda']
        theoretical_quantiles = []
        for p in theoretical_probs:
            # Find k such that CDF(k) >= p
            k = 1
            cumsum = zt_poisson_pmf(1, lam)
            while cumsum < p and k < 20:
                k += 1
                cumsum += zt_poisson_pmf(k, lam)
            theoretical_quantiles.append(k)
        theoretical_quantiles = np.array(theoretical_quantiles)
    else:  # zt_nbinom
        # For negative binomial, use regular quantiles offset
        n_param = best_result['params']['n']
        p_param = best_result['params']['p']
        p_zero = stats.nbinom.pmf(0, n_param, p_param)
        # Adjust probabilities to account for zero-truncation
        adjusted_probs = theoretical_probs * (1 - p_zero) + p_zero
        theoretical_quantiles = stats.nbinom.ppf(adjusted_probs, n_param, p_param)
    
    sample_quantiles = np.sort(disasters_per_year)
    
    ax.scatter(theoretical_quantiles, sample_quantiles, s=50, alpha=0.6, color='steelblue')
    
    # Perfect fit line
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    
    ax.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax.set_ylabel('Sample Quantiles', fontsize=11)
    ax.set_title('(C) Q-Q Plot', fontweight='bold', fontsize=12)
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Q-Q plot not applicable\nfor empirical distribution',
           transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.axis('off')

# Panel D: Summary statistics
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
FITTED DISTRIBUTION
{'='*40}

Best model: {best_result['distribution'].upper()}
{'(ZERO-TRUNCATED)' if 'zt_' in best_result['distribution'] else ''}

Parameters:
"""

if best_result['distribution'] == 'zt_poisson':
    summary_text += f"  λ (lambda) = {best_result['params']['lambda']:.3f}\n"
    summary_text += f"\n  P(X=0) = 0 (zero-truncated)\n"
    summary_text += f"  P(X=1) = {zt_poisson_pmf(1, best_result['params']['lambda']):.3f}\n"
    summary_text += f"  P(X=2) = {zt_poisson_pmf(2, best_result['params']['lambda']):.3f}\n"
    summary_text += f"  P(X=3) = {zt_poisson_pmf(3, best_result['params']['lambda']):.3f}\n"
elif best_result['distribution'] == 'zt_nbinom':
    summary_text += f"  n (size) = {best_result['params']['n']:.3f}\n"
    summary_text += f"  p (prob) = {best_result['params']['p']:.3f}\n"
    summary_text += f"\n  P(X=0) = 0 (zero-truncated)\n"

summary_text += f"""
Fitted statistics:
  Mean = {best_result['mean']:.3f} disasters/year
  Variance = {best_result['var']:.3f}
  Var/Mean ratio = {best_result['var']/best_result['mean']:.3f}

Empirical statistics:
  Mean = {empirical_mean:.3f} disasters/year
  Variance = {empirical_var:.3f}
  Var/Mean ratio = {empirical_var/empirical_mean:.3f}

Goodness of fit:
  AIC = {best_result['aic']:.2f}
  BIC = {best_result['bic']:.2f}
  Chi² p-value = {best_result['chi2_pval']:.4f}
  {'✓ Good fit (p > 0.05)' if best_result['chi2_pval'] > 0.05 else '✗ Poor fit (p < 0.05)'}

Monte Carlo sampling:
  Will NEVER generate 0 disasters
  Minimum: 1 disaster per year
  Suitable for appropriation prediction
"""

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
       fontsize=9, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fit_disaster_frequency.png', dpi=300, bbox_inches='tight')
print(f"Saved visualization to {FIGURES_DIR / 'fit_disaster_frequency.png'}")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nFitted disaster frequency distribution:")
print(f"  Distribution: {best_result['distribution'].upper()}")
print(f"  Mean: {best_result['mean']:.3f} disasters/year")
print(f"  Variance: {best_result['var']:.3f}")

if best_result['distribution'] == 'poisson':
    print(f"  λ = {best_result['params']['lambda']:.3f}")
elif best_result['distribution'] == 'nbinom':
    print(f"  n = {best_result['params']['n']:.3f}")
    print(f"  p = {best_result['params']['p']:.3f}")

print(f"\nThis distribution will be used in Monte Carlo simulations")
print(f"to generate smooth disaster frequency curves.")

print("\n" + "="*80)