"""
Estimating the parameters of representative lorgnormal distribution
using Bayesian Inference (Hierarchical Model)
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Given the observed parameters from individual sample:
shapes = np.array([0.06730584606046047, 0.03234557292014533, 1.2472634953027226e-05, 0.006365498032352482, 0.06080020124353382])
locs = np.array([-1.981507078516348, -30.43220586325637, -131066.56744617317, -223.26101631784076, -8.258912832314465])
scales = np.array([13.339675386889079, 43.412099012735666, 131078.83429217472, 235.2715617511924, 20.542495962045177])

# Compute the individual means and standard deviations of the five lognormals.
# (For reference, these are in the desired range: ~11-13 for the mean and ~0.9-1.6 for the std.)
print("Individual model moments:")
for i, (s_val, loc_val, scale_val) in enumerate(zip(shapes, locs, scales)):
    indiv_mean = lognorm.mean(s_val, loc=loc_val, scale=scale_val)
    indiv_std  = lognorm.std(s_val, loc=loc_val, scale=scale_val)
    print(f"Model {i+1} -- Mean: {indiv_mean:.5f}, Std: {indiv_std:.5f}")

with pm.Model() as hierarchical_model:
    # Use the median (robust estimate) as the initial hyperprior mean to reduce outlier influence.
    med_shape = np.median(shapes)
    med_loc   = np.median(locs)
    med_scale = np.median(scales)
    
    # Hyperpriors for the shape parameter (ensuring positivity)
    mu_shape = pm.Normal('mu_shape', mu=med_shape, sigma=0.05)
    sigma_shape = pm.HalfNormal('sigma_shape', sigma=0.05)
    
    # Hyperpriors for the loc parameter
    mu_loc = pm.Normal('mu_loc', mu=med_loc, sigma=50)
    sigma_loc = pm.HalfNormal('sigma_loc', sigma=50)
    
    # Hyperpriors for the scale parameter (ensuring positivity)
    mu_scale = pm.Normal('mu_scale', mu=med_scale, sigma=50)
    sigma_scale = pm.HalfNormal('sigma_scale', sigma=50)
    
    # Likelihood: Using a robust Student-t likelihood (nu=4) to reduce the influence of extreme outliers.
    shape_obs = pm.StudentT('shape_obs', nu=4, mu=mu_shape, sigma=sigma_shape, observed=shapes)
    loc_obs = pm.StudentT('loc_obs', nu=4, mu=mu_loc, sigma=sigma_loc, observed=locs)
    scale_obs = pm.StudentT('scale_obs', nu=4, mu=mu_scale, sigma=sigma_scale, observed=scales)
    
    # Sample from the posterior (returns an InferenceData object)
    idata = pm.sample(3000, tune=2000, cores=2, target_accept=0.95, random_seed=42)
    # idata = pm.sample(3000, tune=2000, cores=2, target_accept=0.95)
    
    # Print a summary of the posterior distributions for the hyperparameters
    summary = az.summary(idata, hdi_prob=0.95)
    print(summary)

# Extract representative parameters from the posterior using idata.posterior
rep_shape = idata.posterior['mu_shape'].mean().values.item()
rep_loc = idata.posterior['mu_loc'].mean().values.item()
rep_scale = idata.posterior['mu_scale'].mean().values.item()

# Calculate mean and standard deviation of the representative lognormal distribution
rep_mean = lognorm.mean(rep_shape, loc=rep_loc, scale=rep_scale)
rep_std  = lognorm.std(rep_shape, loc=rep_loc, scale=rep_scale)

print("\nRepresentative lognormal parameters:")
print(f"Shape (s): {rep_shape:.5f}")
print(f"Location: {rep_loc:.5f}")
print(f"Scale: {rep_scale:.5f}")
print(f"Mean: {rep_mean:.5f}")
print(f"Std: {rep_std:.5f}")

# Generate and plot the representative lognormal PDF
x = np.linspace(7, 18, 1000)
representative_pdf = lognorm.pdf(x, s=rep_shape, loc=rep_loc, scale=rep_scale)

plt.figure(figsize=(12, 8))
# Plot the original PDFs from the five models for comparison:
for i, (s_val, loc_val, scale_val) in enumerate(zip(shapes, locs, scales)):
    pdf = lognorm.pdf(x, s=s_val, loc=loc_val, scale=scale_val)
    plt.plot(x, pdf, label=f'A{i+1} lognormal fit')
    
# Plot the representative lognormal PDF (dashed black line)
plt.plot(x, representative_pdf, 'k--', label=f'Proposed Lognormal\nMean: {rep_mean:.4f}, Std: {rep_std:.4f}', linewidth=3)

plt.xlabel('Dynamic modulus (GPa)', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Original Lognormal PDFs & Representative PDF')
plt.legend(fontsize=12)
plt.grid(True)
plt.show()