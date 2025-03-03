"""
Estimating the parameters of a representative VARIOGRAM & PLOTTING
Using Bayesian Inference (Hierarchical Bayesian) to select the
best model paremeter from 5 rational quadratic models
"""
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import gstools as gs

# Given the observed parameters from individual sample:
models = [
    gs.Rational(dim=2, var=0.8606, len_scale=1.9877, nugget=False, alpha=0.8007),
    gs.Rational(dim=2, var=2.2718, len_scale=1.5928, nugget=False, alpha=0.5000),
    gs.Rational(dim=2, var=2.7769, len_scale=2.2901, nugget=False, alpha=50.000),
    gs.Rational(dim=2, var=2.5204, len_scale=1.0215, nugget=False, alpha=0.5000),
    gs.Rational(dim=2, var=1.5336, len_scale=1.6095, nugget=False, alpha=50.000)
]

# Extract parameters from models
vars_array = np.array([model.var for model in models])
len_scales_array = np.array([model.len_scale for model in models])
alphas_array = np.array([model.alpha for model in models])

# Print individual model parameters
print("Individual model parameters:")
for i, (var_val, len_scale_val, alpha_val) in enumerate(zip(vars_array, len_scales_array, alphas_array)):
    print(f"Model {i+1} -- Var: {var_val:.5f}, Len_scale: {len_scale_val:.5f}, Alpha: {alpha_val:.5f}")

# Define the rational variogram function
def rational_variogram(h, var, len_scale, alpha):
    """Compute the rational variogram model."""
    h_scaled = h / len_scale
    # return var * (h_scaled**2 / (1.0 + h_scaled**2/alpha))
    return var * (1 - (1 + (1 / alpha) * ((h_scaled) ** 2)) ** (-alpha))

with pm.Model() as hierarchical_model:
    # Use the median as the initial hyperprior mean to reduce outlier influence
    med_var = np.median(vars_array)
    med_len_scale = np.median(len_scales_array)
    
    # Note: Alpha has a bimodal distribution with values around 0.5 and 50
    # We'll handle it differently with a mixture model
    
    # Hyperpriors for variance parameter (ensuring positivity)
    mu_var = pm.Normal('mu_var', mu=med_var, sigma=0.2)
    sigma_var = pm.HalfNormal('sigma_var', sigma=0.2)
    
    # Hyperpriors for length scale parameter (ensuring positivity)
    mu_len_scale = pm.Normal('mu_len_scale', mu=med_len_scale, sigma=0.5)
    sigma_len_scale = pm.HalfNormal('sigma_len_scale', sigma=0.5)
    
    # For alpha, we'll use a mixture model due to its bimodal nature
    # Weight parameter for the mixture (probability of being in the first component)
    w = pm.Beta('w', alpha=1, beta=1)
    
    # First component for small alpha values (~0.5)
    mu_alpha_small = pm.Normal('mu_alpha_small', mu=0.6, sigma=0.2)
    sigma_alpha_small = pm.HalfNormal('sigma_alpha_small', sigma=0.2)
    
    # Second component for large alpha values (~50)
    mu_alpha_large = pm.Normal('mu_alpha_large', mu=50.0, sigma=5.0)
    sigma_alpha_large = pm.HalfNormal('sigma_alpha_large', sigma=5.0)
    
    # Likelihood: Using a robust Student-t likelihood (nu=4) to reduce the influence of extreme outliers
    var_obs = pm.StudentT('var_obs', nu=4, mu=mu_var, sigma=sigma_var, observed=vars_array)
    len_scale_obs = pm.StudentT('len_scale_obs', nu=4, mu=mu_len_scale, sigma=sigma_len_scale, observed=len_scales_array)
    
    # For alpha, we model the likelihood as a mixture of two Student-t distributions
    alpha_small_component = pm.StudentT.dist(nu=4, mu=mu_alpha_small, sigma=sigma_alpha_small)
    alpha_large_component = pm.StudentT.dist(nu=4, mu=mu_alpha_large, sigma=sigma_alpha_large)
    
    # Create the mixture likelihood for alpha
    alpha_mix = pm.Mixture('alpha_mix', w=[w, 1-w], 
                           comp_dists=[alpha_small_component, alpha_large_component],
                           observed=alphas_array)
    
    # Sample from the posterior
    # idata = pm.sample(1000, tune=500, cores=2, target_accept=0.95, random_seed=42)
    idata = pm.sample(1000, tune=500, cores=2, target_accept=0.95)
    
    # Print a summary of the posterior distributions for the hyperparameters
    summary = az.summary(idata, hdi_prob=0.95)
    print("\nPosterior Summary:")
    print(summary)

# Extract posterior samples
posterior = idata.posterior

# Calculate the probability of being in each mixture component
p_small = posterior['w'].mean().values.item()
p_large = 1 - p_small

print(f"\nProbability of alpha being in small component (~0.5): {p_small:.3f}")
print(f"Probability of alpha being in large component (~50): {p_large:.3f}")

# Determine which component is more likely
more_likely_component = "small" if p_small > p_large else "large"

# Extract representative parameters from the posterior
rep_var = posterior['mu_var'].mean().values.item()
rep_len_scale = posterior['mu_len_scale'].mean().values.item()

# For alpha, choose the more likely component based on the mixture weights
if more_likely_component == "small":
    rep_alpha = posterior['mu_alpha_small'].mean().values.item()
else:
    rep_alpha = posterior['mu_alpha_large'].mean().values.item()

print("\nRepresentative variogram parameters:")
print(f"Variance (var): {rep_var:.5f}")
print(f"Length scale: {rep_len_scale:.5f}")
print(f"Alpha: {rep_alpha:.5f}")
print(f"Alpha component used: {more_likely_component}")

# Define the optimized model using the representative parameters
proposed_model = gs.Rational(dim=2, var=rep_var, len_scale=rep_len_scale, nugget=False, alpha=rep_alpha)

# Plotting the models
x = np.linspace(0, 42, 100)

plt.figure(figsize=(12, 8))

# Plot original models
for i, model in enumerate(models):
    plt.plot(x, model.variogram(x), label=f'A{i+1}')

# Plot the proposed model
proposed_label = f'Proposed model\nvar={rep_var:.2f}\nlen_scale={rep_len_scale:.2f}\nalpha={rep_alpha:.2f}'
plt.plot(x, proposed_model.variogram(x), 'k--', linewidth=3, label=proposed_label)

# Customize the plot
plt.legend(loc='lower right')
plt.xlabel('Lag distance ($\\Delta s$-cm)', fontsize=12)
plt.ylabel(r'Gamma $\gamma(h)$', fontsize=12)
plt.yticks(np.arange(0, 3.12, 0.25), fontsize=12)
plt.xticks(np.arange(0, 41, 5), fontsize=12)
plt.grid(True)

plt.show()