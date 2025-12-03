'''
Updated model fitting script with posterior saving for Monte Carlo simulations.
'''

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import cmdstanpy
import numpy as np
import sys

# Add the utils to path if needed
sys.path.append(str(Path(__file__).parent))

#cmdstanpy.install_cmdstan()
#cmdstanpy.install_cmdstan(compiler=True)

CSV_DIR = Path('data/csv')
OUTPUT_DIR = Path('data/output')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
STAN_MODEL_PATH = Path('src/api/model.stan')

print("="*80)
print("BAYESIAN MODEL FITTING FOR DISASTER APPROPRIATIONS")
print("="*80)

# ==============================================================================
# 1. Load and prepare data
# ==============================================================================
print("\nLoading data...")
df = pd.read_csv(CSV_DIR / 'model_data.csv')

print(f"Initial rows: {len(df)}")

# Drop missing values
df = df.dropna(subset=[
    'log_appropriation',
    'log_total_value',
    'peak_intensity',
    'duration_days',
    'percent_in_heatwave'
])

# Filter to valid values
df = df[(df['peak_intensity'] > 0) & (df['log_appropriation'] > 0)]

print(f'Total rows after dropping NAs and filtering: {len(df)}')

# ==============================================================================
# 2. Prepare data for Stan
# ==============================================================================
print("\nPreparing data for Stan...")

# Calculate scaling parameters (save these for Monte Carlo)
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

# Save scaling parameters for use in Monte Carlo
scaling_df = pd.DataFrame([scaling_params])
scaling_df.to_csv(OUTPUT_DIR / 'scaling_parameters.csv', index=False)
print(f"Saved scaling parameters to {OUTPUT_DIR / 'scaling_parameters.csv'}")

# Prepare Stan data
bayesian_data = {
    'N': len(df),
    'y': df['log_appropriation'].values,
    'x_value': (df['log_total_value'] - scaling_params['log_value_mean']) / scaling_params['log_value_std'],
    'x_peak': (df['peak_intensity'] - scaling_params['peak_mean']) / scaling_params['peak_std'],
    'x_duration': (df['duration_days'] - scaling_params['duration_mean']) / scaling_params['duration_std'],
    'x_percent': (df['percent_in_heatwave'] - scaling_params['percent_mean']) / scaling_params['percent_std']
}

print(f"\nStan data prepared:")
print(f"  N = {bayesian_data['N']}")
print(f"  Predictors standardized (mean=0, sd=1)")

# ==============================================================================
# 3. Compile and fit model
# ==============================================================================
print("\n" + "="*80)
print("FITTING BAYESIAN MODEL")
print("="*80)

print("\nCompiling Stan model...")
bayesian_model = cmdstanpy.CmdStanModel(stan_file=STAN_MODEL_PATH)

print("Running MCMC sampling...")
fit = bayesian_model.sample(
    data=bayesian_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    show_progress=True
)

print("\nSampling complete!")

# ==============================================================================
# 4. Save results
# ==============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save summary statistics
print("\nSaving summary statistics...")
summary = fit.summary()
summary.to_csv(OUTPUT_DIR / 'bayesian_model_summary.csv')
print(f"Saved to {OUTPUT_DIR / 'bayesian_model_summary.csv'}")

# Save full posterior samples for Monte Carlo
print("\nSaving full posterior samples...")
posterior_df = fit.draws_pd()
posterior_df.to_csv(OUTPUT_DIR / 'posterior_samples.csv', index=False)
print(f"Saved {len(posterior_df)} posterior samples to {OUTPUT_DIR / 'posterior_samples.csv'}")
print(f"  (These will be used for Monte Carlo simulations)")

# Save diagnostics
print("\nSaving MCMC diagnostics...")
diagnostics = fit.diagnose()
with open(OUTPUT_DIR / 'mcmc_diagnostics.txt', 'w') as f:
    f.write(diagnostics)
print(f"Saved to {OUTPUT_DIR / 'mcmc_diagnostics.txt'}")

# ==============================================================================
# 5. Print key results
# ==============================================================================
print("\n" + "="*80)
print("MODEL RESULTS")
print("="*80)

print("\nPosterior estimates (on standardized scale):")
# Filter to show only key parameters (not log_lik or y_pred)
key_params = ['lp__', 'alpha', 'beta_value', 'beta_peak', 'beta_duration', 'beta_percent', 'sigma']
summary_filtered = summary.loc[key_params]

# Handle different CmdStanPy versions
if 'ESS_bulk' in summary.columns:
    # Newer version with ESS_bulk/ESS_tail
    print(summary_filtered[['Mean', 'StdDev', '5%', '95%', 'ESS_bulk', 'R_hat']])
elif 'ESS' in summary.columns:
    # Mid version with ESS
    print(summary_filtered[['Mean', 'StdDev', '5%', '95%', 'ESS', 'R_hat']])
elif 'N_Eff' in summary.columns:
    # Older version with N_Eff
    print(summary_filtered[['Mean', 'StdDev', '5%', '95%', 'N_Eff', 'R_hat']])
else:
    # Just print what we have
    print(summary_filtered)

# Calculate effects on original scale
print("\n" + "="*80)
print("INTERPRETATION (effect of 1 SD increase in predictor)")
print("="*80)

beta_value_mean = summary.loc['beta_value', 'Mean']
beta_peak_mean = summary.loc['beta_peak', 'Mean']
beta_duration_mean = summary.loc['beta_duration', 'Mean']
beta_percent_mean = summary.loc['beta_percent', 'Mean']

print(f"\n1 SD increase in log(value) → {np.exp(beta_value_mean):.2f}x appropriation")
print(f"1 SD increase in peak intensity → {np.exp(beta_peak_mean):.2f}x appropriation")
print(f"1 SD increase in duration → {np.exp(beta_duration_mean):.2f}x appropriation")
print(f"1 SD increase in % in heatwave → {np.exp(beta_percent_mean):.2f}x appropriation")

# ==============================================================================
# 6. Model diagnostics
# ==============================================================================
print("\n" + "="*80)
print("MODEL DIAGNOSTICS")
print("="*80)

# Check R-hat (handle different column names)
rhat_col = 'R_hat' if 'R_hat' in summary.columns else 'Rhat'
max_rhat = summary[rhat_col].max()
print(f"\nMaximum R-hat: {max_rhat:.4f}")
if max_rhat < 1.01:
    print("  ✓ All chains converged (R-hat < 1.01)")
elif max_rhat < 1.05:
    print("  ⚠ Chains mostly converged (R-hat < 1.05)")
else:
    print("  ✗ Convergence issues (R-hat > 1.05)")

# Check effective sample size (handle different column names)
if 'ESS_bulk' in summary.columns:
    ess_col = 'ESS_bulk'
elif 'ESS' in summary.columns:
    ess_col = 'ESS'
else:
    ess_col = 'N_Eff'
min_neff = summary[ess_col].min()
print(f"\nMinimum effective sample size: {min_neff:.0f}")
if min_neff > 400:
    print("  ✓ Good effective sample size (ESS > 400)")
elif min_neff > 100:
    print("  ⚠ Adequate effective sample size (ESS > 100)")
else:
    print("  ✗ Low effective sample size (ESS < 100)")

# ==============================================================================
# 7. Comparison with OLS
# ==============================================================================
print("\n" + "="*80)
print("COMPARISON WITH OLS")
print("="*80)

from sklearn.linear_model import LinearRegression

# Prepare data for OLS
X = np.column_stack([
    bayesian_data['x_value'],
    bayesian_data['x_peak'],
    bayesian_data['x_duration'],
    bayesian_data['x_percent']
])
y = bayesian_data['y']

# Fit OLS
ols_model = LinearRegression()
ols_model.fit(X, y)

print("\nOLS coefficients:")
print(f"  Intercept: {ols_model.intercept_:.4f}")
print(f"  log(value): {ols_model.coef_[0]:.4f}")
print(f"  Peak intensity: {ols_model.coef_[1]:.4f}")
print(f"  Duration: {ols_model.coef_[2]:.4f}")
print(f"  Percent in heatwave: {ols_model.coef_[3]:.4f}")

print("\nBayesian posterior means:")
print(f"  Intercept: {summary.loc['alpha', 'Mean']:.4f}")
print(f"  log(value): {summary.loc['beta_value', 'Mean']:.4f}")
print(f"  Peak intensity: {summary.loc['beta_peak', 'Mean']:.4f}")
print(f"  Duration: {summary.loc['beta_duration', 'Mean']:.4f}")
print(f"  Percent in heatwave: {summary.loc['beta_percent', 'Mean']:.4f}")

print("\n(Bayesian estimates should be similar to OLS with weak priors)")

# ==============================================================================
# 8. Next steps
# ==============================================================================
print("\n" + "="*80)
print("MODEL FITTING COMPLETE!")
print("="*80)

print("\nFiles saved:")
print(f"  ✓ data/csv/bayesian_model_summary.csv")
print(f"  ✓ {OUTPUT_DIR / 'posterior_samples.csv'}")
print(f"  ✓ {OUTPUT_DIR / 'scaling_parameters.csv'}")
print(f"  ✓ {OUTPUT_DIR / 'mcmc_diagnostics.txt'}")

print("\nNext steps:")
print("  1. Review MCMC diagnostics (check for convergence)")
print("  2. Run monte_carlo_simulation.py for baseline analysis")
print("  3. Run stan_posterior_utils.py for full Bayesian Monte Carlo")
print("  4. Check MONTE_CARLO_GUIDE.md for detailed instructions")

print("\nQuick command to run Monte Carlo:")
print("  python monte_carlo_simulation.py")