# Constrained regret minimization using Con-LCB

We demonstrate our algorithm Con-LCB on two examples. The first instance is a four-armed bandit problem
with the arms distributed as two-dimensional Gaussians. The aim is to minimize the mean of the
first dimension constrained on the mean of the second dimension. The second instance is also a four-armed 
bandit problem with Beta distributed arms. The aim is to minimize the mean of the arms subject to a 
constraint on the variance of each arm. We also show that Lagrangian relaxations for constrained bandit
problems can be highly suboptimal because it is difficult to choose the 'correct' multiplier for the 
problem. 

### Prerequisites

The code was run on a system with the following:

Python 3.7.6
Numpy 1.18.1
Scipy 1.4.1

### Usage

The directory has 3 files. The file 'gaussian.py' demonstrates Con-LCB
on the Gaussian instance described before, the file 'mv_beta.py' 
demonstrates Con-LCB on the Beta-distributed instance described before,
and the file 'lag_lcb_beta.py' demonstrates the Lagrangian relaxation
on the Beta-distributed instance mentioned above. 