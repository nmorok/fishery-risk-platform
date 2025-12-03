data {
  int<lower=1> N;
  vector[N] y;
  vector[N] x_value;
  vector[N] x_peak;
  vector[N] x_duration;
  vector[N] x_percent;
}

parameters {
  real alpha;
  real beta_value;
  real beta_peak;
  real beta_duration;
  real beta_percent;
  real<lower=0> sigma;
}

model {
  alpha ~ normal(16, 3);
  beta_value ~ normal(0, 2);
  beta_peak ~ normal(0, 1);
  beta_duration ~ normal(0, 1);
  beta_percent ~ normal(0, 1);
  sigma ~ exponential(1);
  
  y ~ normal(alpha + beta_value * x_value + beta_peak * x_peak +
             beta_duration * x_duration + beta_percent * x_percent, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_pred;
  
  for (i in 1:N) {
    real mu = alpha + beta_value * x_value[i] + beta_peak * x_peak[i] +
              beta_duration * x_duration[i] + beta_percent * x_percent[i];
    log_lik[i] = normal_lpdf(y[i] | mu, sigma);
    y_pred[i] = normal_rng(mu, sigma);
  }
}