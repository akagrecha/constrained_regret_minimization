import numpy as np

# 2D Gaussian arms

def data_function(arm):
	"""
	Parameters:
		arm: int,
			index of arm to be sampled
	Returns:
		sample: float
			sample from the specified arm
	"""
	arm0_means = np.array([0.3, 0.4, 0.2, 0.5])
	arm1_means = np.array([0.4, 0.3, 1.0, 1.0])
	mean = [arm0_means[arm], arm1_means[arm]]
	cov = [[1,0.5], [0.5, 1]]
	sample = np.random.multivariate_normal(mean, cov)
	return sample

def con_lcb(T, K, threshold):
	"""
	Parameters:
		T: int, 
			time horizon 
		K: int
			number of arms
		threshold: float
			threshold for mean of second dimension 
	Returns:
		out: np.ndarray
			array consisting of number of pulls of each arm and
			the feasibility flag
	"""
	arm0_means = np.zeros(K)
	arm1_means = np.zeros(K)
	num_pulls = np.zeros(K)
	
	for i in range(K):
		samples = np.array([data_function(i) for j in range(1)])
		arm0_means[i] = np.mean(samples[:,0])
		arm1_means[i] = np.mean(samples[:,1])
		num_pulls[i] = 1
	
	for t in range(T-K):
		plausible_arr = np.array([i for i in range(K) if 
				  arm1_means[i] <=  threshold + ((np.log(2)+2*np.log(T))/(0.5*num_pulls[i]))**0.5])
		if len(plausible_arr) > 0:
			arm = plausible_arr[np.argmin([arm0_means[i] - ((np.log(2)+2*np.log(T))/(0.5*num_pulls[i]))**0.5 
							 for i in plausible_arr])]
		else:
			#print("problem")
			arm = np.argmin([arm1_means[i] - ((np.log(2)+2*np.log(T))/(0.5*num_pulls[i]))**0.5 
							 for i in range(K)])
		sample = data_function(arm)
		arm0_means[arm] = (sample[0] + num_pulls[arm]*arm0_means[arm])/(num_pulls[arm]+1)
		arm1_means[arm] = (sample[1] + num_pulls[arm]*arm1_means[arm])/(num_pulls[arm]+1)
		num_pulls[arm] += 1
	
	plausible_arr = np.array([i for i in range(K) if 
				  arm1_means[i] <=  threshold + ((np.log(2)+2*np.log(T))/(0.5*num_pulls[i]))**0.5])
	feasibility_flag = 0
	if len(plausible_arr)>0:
		feasibility_flag = 1
	return np.append(num_pulls, feasibility_flag)
