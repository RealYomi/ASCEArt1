
"""
Represenative PDF
Using Bayesian Inference(Single level) to select the
best model paremeter from 5 rational quadratic models
"""
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy import stats
from scipy.stats import lognorm

# Given parameters from your five lognormal models:
shapes = np.array([0.06730584606046047, 0.03234557292014533, 1.2472634953027226e-05, 0.006365498032352482, 0.06080020124353382])
locs   = np.array([-1.981507078516348, -30.43220586325637, -131066.56744617317, -223.26101631784076, -8.258912832314465])
scales = np.array([13.339675386889079, 43.412099012735666, 131078.83429217472, 235.2715617511924, 20.542495962045177])

# Choose a plotting range based on the 5th and 95th percentiles of the models:
p05 = np.array([stats.lognorm.ppf(0.05, s, loc=l, scale=sc) for s, l, sc in zip(shapes, locs, scales)])
p95 = np.array([stats.lognorm.ppf(0.95, s, loc=l, scale=sc) for s, l, sc in zip(shapes, locs, scales)])
min_x = np.min(p05)
max_x = np.max(p95)
x = np.linspace(min_x, max_x, 100)

# Compute PDFs for each of the five lognormal models using SciPy (for synthetic data)
pdfs = np.array([stats.lognorm.pdf(x, s, loc=l, scale=sc) for s, l, sc in zip(shapes, locs, scales)])
avg_pdf = np.mean(pdfs, axis=0)

# Create synthetic observations by adding a small amount of noise to the averaged PDF
np.random.seed(123)
obs_noise = 0.05 * np.max(avg_pdf)
observed_pdf = avg_pdf + np.random.normal(0, obs_noise, size=len(avg_pdf))

# Define a symbolic lognormal PDF using Aesara operations
def symbolic_lognormal_pdf(x, shape, loc, scale):
    """
    Compute the lognormal PDF symbolically.
    The SciPy lognorm.pdf is defined as:
      pdf(x; s, loc, scale) = 1/((x - loc)*s*sqrt(2*pi)) * exp(- (log((x - loc)/scale))**2 / (2*s**2))
    Note: x is assumed to be greater than loc.
    """
    return 1/((x - loc)*shape*np.sqrt(2*np.pi)) * np.exp(- (np.log((x - loc)/scale))**2 / (2*shape**2))

# Reparameterized model with log-scale for positive parameters and standardized priors
with pm.Model() as reparam_model:
    # Log-scale priors for shape (avoids small numbers that cause computational issues)
    log_shape = pm.Normal('log_shape', 
                         mu=np.log(np.median(shapes)), 
                         sigma=1.0)
    shape_param = pm.Deterministic('shape', pm.math.exp(log_shape))
    
    # For location, use standardized parameterization
    loc_median = np.median(locs)
    loc_mad = np.median(np.abs(locs - loc_median)) * 1.4826
    loc_param = pm.Normal('loc', mu=loc_median, sigma=loc_mad * 2)
    
    # Log-scale prior for scale parameter
    log_scale = pm.Normal('log_scale', 
                         mu=np.log(np.median(scales)), 
                         sigma=1.0)
    scale_param = pm.Deterministic('scale', pm.math.exp(log_scale))
    
    # Noise parameter
    sigma_noise = pm.HalfNormal('sigma_noise', sigma=0.05 * np.max(avg_pdf))
    
    # Use precomputed values where possible to reduce computational burden
    predicted_pdf = pm.Deterministic('predicted_pdf', 
                                   symbolic_lognormal_pdf(x, shape_param, loc_param, scale_param))
    
    # Student's T likelihood for robustness
    likelihood = pm.StudentT('likelihood', nu=4, mu=predicted_pdf, sigma=sigma_noise, observed=observed_pdf)

    # Sample from the posterior
    trace = pm.sample(1000, tune=500, return_inferencedata=True, chains=4, cores=2, target_accept=0.95)
    
    # Perform MAP estimation as well
    map_estimate = pm.find_MAP()

# Extract posterior samples
posterior_samples = az.extract(trace, var_names=['shape', 'loc', 'scale', 'sigma_noise'])
shape_samples = posterior_samples['shape'].values
loc_samples = posterior_samples['loc'].values
scale_samples = posterior_samples['scale'].values

post_mean_shape = posterior_samples['shape'].mean().values
post_mean_loc = posterior_samples['loc'].mean().values
post_mean_scale = posterior_samples['scale'].mean().values

# Calculate mean and standard deviation of the lognormal distribution
mean_lognormal = lognorm.mean(post_mean_shape, loc=post_mean_loc, scale=post_mean_scale)
std_lognormal = lognorm.std(post_mean_shape, loc=post_mean_loc, scale=post_mean_scale)

print(f"Mean of the lognormal distribution: {mean_lognormal:.4f}")
print(f"Standard deviation of the lognormal distribution: {std_lognormal:.4f}")

# Function to generate predictions with uncertainty intervals
def predict_pdf_with_uncertainty(x, samples, n_samples=100):
    """Predict PDF values with uncertainty intervals."""
    # Randomly select a subset of posterior samples
    idx = np.random.choice(len(samples['shape']), size=n_samples, replace=False)
    # Generate predictions for each sampled parameter set
    predictions = np.zeros((n_samples, len(x)))
    for i in range(n_samples):
        # Compute the PDF using SciPy for each sampled parameter set (for plotting only)
        predictions[i, :] = stats.lognorm.pdf(x, samples['shape'][idx[i]], loc=samples['loc'][idx[i]], scale=samples['scale'][idx[i]])
    mean_pred = np.mean(predictions, axis=0)
    lower_ci = np.percentile(predictions, 2.5, axis=0)
    upper_ci = np.percentile(predictions, 97.5, axis=0)
    return mean_pred, lower_ci, upper_ci

# Prepare dictionary of samples and generate predictions with uncertainty
samples_dict = {'shape': shape_samples, 'loc': loc_samples, 'scale': scale_samples}
mean_pred, lower_ci, upper_ci = predict_pdf_with_uncertainty(x, samples_dict)

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot the five original lognormal PDFs
for i, (s, l, sc) in enumerate(zip(shapes, locs, scales)):
    plt.plot(x, stats.lognorm.pdf(x, s, loc=l, scale=sc), label=f'A{i+1} lognormal fit')

# Plot the Bayesian representative lognormal PDF with its 95% credible interval
plt.plot(x, mean_pred, 'k--', linewidth=3,
         label=f'Bayesian model(Proposed model)\n(shape={post_mean_shape:.2f}, loc={post_mean_loc:.2f}, scale={post_mean_scale:.2f})\n(Mean={mean_lognormal:.4f}, Std={std_lognormal:.4f})')
plt.fill_between(x, lower_ci, upper_ci, color='red', alpha=0.2, label='95% credible interval')

# Plot the synthetic observations
plt.scatter(x[::10], observed_pdf[::10], color='black', s=30, alpha=0.6, label='Observations')

# Customize plot
plt.xlabel('Dynamic modulus (GPa)', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=12)
plt.title('Bayesian Representative Lognormal Distribution with Uncertainty')
plt.grid(True)

plt.show()

# Plot posterior traces for inspection
az.plot_trace(trace, var_names=['shape', 'loc', 'scale', 'sigma_noise'])
plt.tight_layout()

# Plot posterior densities in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
az.plot_posterior(trace, var_names=['shape'], ax=axes[0, 0])
az.plot_posterior(trace, var_names=['loc'], ax=axes[0, 1])
az.plot_posterior(trace, var_names=['scale'], ax=axes[1, 0])
az.plot_posterior(trace, var_names=['sigma_noise'], ax=axes[1, 1])
plt.tight_layout()

# Print out model results
print("Bayesian Lognormal Model Results:")
print(f"Posterior mean shape: {post_mean_shape:.4f}")
print(f"Posterior mean loc:   {post_mean_loc:.4f}")
print(f"Posterior mean scale: {post_mean_scale:.4f}")
print("\nMAP Estimates:")
print(f"MAP shape: {map_estimate['shape']:.4f}")
print(f"MAP loc:   {map_estimate['loc']:.4f}")
print(f"MAP scale: {map_estimate['scale']:.4f}")

# Optionally, add pairwise joint posterior plots
az.plot_pair(trace, var_names=['shape', 'loc', 'scale'], kind='kde', marginals=True, figsize=(10, 10))
plt.tight_layout()

plt.show()

# Compare distributions using KL divergence
def calculate_kl_divergence(dist1, dist2, x):
    """Calculate KL divergence between two distributions."""
    # Ensure non-zero probabilities by adding a small constant
    dist1 = dist1 + 1e-10
    dist2 = dist2 + 1e-10
    
    # Normalize
    dist1 = dist1 / np.trapz(dist1, x)
    dist2 = dist2 / np.trapz(dist2, x)
    
    # Calculate KL divergence
    kl_div = np.trapz(dist1 * np.log(dist1 / dist2), x)
    return kl_div

# Calculate KL divergence between each original model and the Bayesian model
print("\nKL Divergence from Bayesian model to original models:")
for i in range(len(shapes)):
    kl_div = calculate_kl_divergence(pdfs[i], mean_pred, x)
    print(f"Model {i+1}: {kl_div:.6f}")

# Calculate KL divergence from average PDF to Bayesian model
kl_div_avg = calculate_kl_divergence(avg_pdf, mean_pred, x)
print(f"Average PDF: {kl_div_avg:.6f}")

plt.show()