"""
Estimating the parameters of a representative VARIOGRAM & PLOTTING
Using Bayesian Inference (Standard single level) to select the
best model paremeter from 5 rational quadratic models
"""

import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy import stats
import pandas as pd
import seaborn as sns

# Define the data from existing models
models = [
    gs.Rational(dim=2, var=0.8606, len_scale=1.9877, nugget=False, alpha=0.8007),
    gs.Rational(dim=2, var=2.2718, len_scale=1.5928, nugget=False, alpha=0.5000),
    gs.Rational(dim=2, var=2.7769, len_scale=2.2901, nugget=False, alpha=50.000),
    gs.Rational(dim=2, var=2.5204, len_scale=1.0215, nugget=False, alpha=0.5000),
    gs.Rational(dim=2, var=1.5336, len_scale=1.6095, nugget=False, alpha=50.000)
]

# Generate synthetic data points for demonstration
x = np.linspace(0, 42, 100)
variograms = np.array([model.variogram(x) for model in models])
avg_variogram = np.mean(variograms, axis=0)

# Add some noise to create synthetic observations
np.random.seed(123)
obs_noise = 0.05
observed_variogram = avg_variogram + np.random.normal(0, obs_noise, size=len(avg_variogram))

# Function to compute rational variogram model
def rational_variogram(x, var, len_scale, alpha):
    """Compute the rational variogram model."""
    h = np.array(x)
    # Handle the case where h=0 to avoid division by zero
    h_scaled = h / len_scale
    # return var * (h_scaled**2 / (1.0 + h_scaled**2/alpha))
    return var * (1 - (1 + (1 / alpha) * ((h_scaled) ** 2)) ** (-alpha))


# Define the Bayesian model
with pm.Model() as model:
    # Prior distributions based on the existing models
    var = pm.TruncatedNormal('var', 
                          mu=np.mean([m.var for m in models]),
                          sigma=np.std([m.var for m in models]) * 2,
                          lower=0.1)
    
    len_scale = pm.TruncatedNormal('len_scale', 
                                mu=np.mean([m.len_scale for m in models]),
                                sigma=np.std([m.len_scale for m in models]) * 2,
                                lower=0.1)
    
    # For alpha, we observe bimodal behavior (values near 0.5 and 50)
    # Using a mixture model to capture this
    w = pm.Beta('w', alpha=1, beta=1)  # Weight parameter for mixture
    alpha_comp1 = pm.TruncatedNormal('alpha_comp1', mu=0.6, sigma=0.2, lower=0.5, upper=1.0)
    alpha_comp2 = pm.TruncatedNormal('alpha_comp2', mu=40, sigma=15, lower=10, upper=50)
    alpha = pm.Deterministic('alpha', w * alpha_comp1 + (1 - w) * alpha_comp2)
    
    # Noise parameter
    sigma = pm.HalfNormal('sigma', sigma=0.1)
    
    # Likelihood function
    predicted = pm.Deterministic('predicted', rational_variogram(x, var, len_scale, alpha))
    likelihood = pm.Normal('likelihood', 
                        mu=predicted, 
                        sigma=sigma, 
                        observed=observed_variogram)
    
    # Sample from the posterior
    trace = pm.sample(1000, tune=500, return_inferencedata=True, 
                    chains=4, cores=1, target_accept=0.95)
    
    # Also perform MAP estimation
    map_estimate = pm.find_MAP()

# Extract posterior samples
posterior_samples = az.extract(trace, var_names=['var', 'len_scale', 'alpha', 'sigma'])
var_samples = posterior_samples['var'].values
len_scale_samples = posterior_samples['len_scale'].values
alpha_samples = posterior_samples['alpha'].values

# Calculate posterior means
post_mean_var = posterior_samples['var'].mean().values
post_mean_len_scale = posterior_samples['len_scale'].mean().values
post_mean_alpha = posterior_samples['alpha'].mean().values

# Function to calculate the predicted variogram with uncertainty
def predict_with_uncertainty(x, samples, n_samples=100):
    """Predict variogram values with uncertainty intervals."""
    # Randomly select a subset of posterior samples
    idx = np.random.choice(len(samples['var']), size=n_samples, replace=False)
    
    # Generate predictions for each sampled parameter set
    predictions = np.zeros((n_samples, len(x)))
    for i in range(n_samples):
        predictions[i, :] = rational_variogram(
            x, samples['var'][idx[i]], samples['len_scale'][idx[i]], samples['alpha'][idx[i]]
        )
    
    # Calculate mean and credible intervals
    mean_pred = np.mean(predictions, axis=0)
    lower_ci = np.percentile(predictions, 2.5, axis=0)
    upper_ci = np.percentile(predictions, 97.5, axis=0)
    
    return mean_pred, lower_ci, upper_ci

# Prepare samples dictionary
samples = {
    'var': var_samples,
    'len_scale': len_scale_samples,
    'alpha': alpha_samples
}

# Generate predictions with uncertainty
mean_pred, lower_ci, upper_ci = predict_with_uncertainty(x, samples)

# Plot the results
plt.figure(figsize=(12, 8))

# # Plot original models
for i, model in enumerate(models):
    plt.plot(x, model.variogram(x), alpha=0.5, label=f'A{i+1}')

# Plot synthetic observations
plt.scatter(x[::5], observed_variogram[::5], color='black', s=30, alpha=0.6, label='Observations')

# Plot Bayesian model predictions with uncertainty
plt.plot(x, mean_pred, 'k--', linewidth=3, label=f'Bayesian model (Proposed model)\n(var={post_mean_var:.2f}, len_scale={post_mean_len_scale:.2f}, alpha={post_mean_alpha:.2f})')
plt.fill_between(x, lower_ci, upper_ci, color='red', alpha=0.2, label='95% credible interval')

# Customize plot
plt.xlabel('Lag distance ($\\Delta s$-cm)', fontsize=12)
plt.ylabel(r'Gamma $\gamma(h)$', fontsize=12)
plt.grid(True)
plt.yticks(np.arange(0, 3.12, 0.25))
plt.xticks(np.arange(0, 41, 5))
plt.legend(loc='lower right')
plt.title('Bayesian Rational Quadratic Variogram Model with Uncertainty')

# Create posterior plots
az.plot_trace(trace, var_names=['var', 'len_scale', 'alpha', 'sigma'])
plt.tight_layout()

# Plot posterior densities
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
az.plot_posterior(trace, var_names=['var'], ax=axes[0, 0])
az.plot_posterior(trace, var_names=['len_scale'], ax=axes[0, 1])
az.plot_posterior(trace, var_names=['alpha'], ax=axes[1, 0])
az.plot_posterior(trace, var_names=['sigma'], ax=axes[1, 1])
plt.tight_layout()

# Print the model comparison metrics
print("Bayesian Model Results:")
print(f"Posterior mean variance: {post_mean_var:.4f}")
print(f"Posterior mean length scale: {post_mean_len_scale:.4f}")
print(f"Posterior mean alpha: {post_mean_alpha:.4f}")
print("\nMAP Estimates:")
print(f"MAP variance: {map_estimate['var']:.4f}")
print(f"MAP length scale: {map_estimate['len_scale']:.4f}")
print(f"MAP alpha: {map_estimate['alpha']:.4f}")

# Calculate model comparison metrics
def compute_waic_and_loo():
    with pm.Model() as model:
        var = pm.TruncatedNormal('var', mu=post_mean_var, sigma=0.1, lower=0.1)
        len_scale = pm.TruncatedNormal('len_scale', mu=post_mean_len_scale, sigma=0.1, lower=0.1)
        alpha = pm.TruncatedNormal('alpha', mu=post_mean_alpha, sigma=0.1, lower=0.1)
        sigma = pm.HalfNormal('sigma', sigma=0.1)
        
        predicted = pm.Deterministic('predicted', rational_variogram(x, var, len_scale, alpha))
        likelihood = pm.Normal('likelihood', mu=predicted, sigma=sigma, observed=observed_variogram)
        
        # Calculate WAIC and LOO
        waic = az.waic(trace, model=model)
        loo = az.loo(trace, model=model)
        return waic, loo

# compute model comparison metrics
waic, loo = compute_waic_and_loo()
print("\nModel Comparison Metrics:")
print(f"WAIC: {waic}")
print(f"LOO: {loo}")

# Add pairwise joint posterior plots
az.plot_pair(trace, var_names=['var', 'len_scale', 'alpha'], 
           kind='kde', marginals=True, figsize=(10, 10))
plt.tight_layout()

plt.show()