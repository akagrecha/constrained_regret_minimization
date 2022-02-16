import numpy as np
import scipy.stats as ss

# Lagrangian relaxation for variance constrained mean optimization

arm_means = np.array([0.3, 0.4, 0.2, 0.5])
arm_vars = np.array([0.08, 0.06, 0.15, 0.15])
alphas = (arm_means*(1-arm_means)/arm_vars -1)*arm_means
betas = (arm_means*(1-arm_means)/arm_vars -1)*(1-arm_means)
def data_function(arm):
	"""
	Parameters:
		arm: int,
			index of arm to be sampled
	Returns:
		sample: float
			sample from the specified arm
	"""
	sample = ss.beta(alphas[arm], betas[arm]).rvs()
	return sample

def find_sample_var(var_prev, mean_prev, sample, size_prev):
	"""
	Parameters:
		var_prev: float, 
			sample variance of samples observed till previous timestep 
		mean_prev: float
			sample mean of samples observed till previous timestep 
		sample: float
			new sample
		size_prev: int
			number of samples observed till previous timestep 
	Returns:
		nv: float
			sample variance of all the samples observed till now
	"""
	nv = (size_prev-1)*var_prev/size_prev + (sample-mean_prev)**2/(size_prev+1)
	return nv

def lagrangian_lcb(T, K, multiplier):
	"""
	Parameters:
		T: int, 
			time horizon 
		K: int
			number of arms
		multiplier: float
			Lagrange multiplier for variance 
	Returns:
		out: np.ndarray
			array consisting of number of pulls of each arm 
	"""
	sample_means = np.zeros(K)
	sample_vars = np.zeros(K)
	num_pulls = np.zeros(K)
	mult_const = (np.log(2) + 2*np.log(T))**0.5
	
	for i in range(K):
		samples = np.array([data_function(i) for j in range(2)])
		sample_means[i] = np.mean(samples)
		sample_vars[i] = np.var(samples)*2/(2-1)
		num_pulls[i] = 2
	
	for t in range(T-2*K):
		lcb_lag = sample_means + multiplier*sample_vars - (1+multiplier)*mult_const/(0.5*num_pulls)**0.5
		
		arm = np.argmin(lcb_lag)
		
		sample = data_function(arm)
		sample_vars[arm] = find_sample_var(sample_vars[arm], sample_means[arm], 
											sample, num_pulls[arm])
		sample_means[arm] = (sample + num_pulls[arm]*sample_means[arm])/(num_pulls[arm]+1)
		num_pulls[arm] += 1
	
	return num_pulls