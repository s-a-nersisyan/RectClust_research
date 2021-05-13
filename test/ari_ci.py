# Author: Benedikt Boecking
# License: MIT
# The ari computation was adapted from sklearn
# The CI computation follows 
# Steinley, Douglas, Michael J. Brusco, and Lawrence Hubert. "The variance of the adjusted Rand index." 
#    Psychological methods 21.2 (2016): 261.


import numpy as np
from scipy.special import comb as spcomb
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import norm


def ari_ci(clusters,trueclusters,alpha = 0.05):
    '''
    
    Steinley, Douglas, Michael J. Brusco, and Lawrence Hubert. "The variance of the adjusted Rand index." 
    Psychological methods 21.2 (2016): 261.
    
    Parameters
    ----------
    clusters : array, shape = [n_samples]
        Cluster labels to evaluate
    trueclusters : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    alpha : float
        Alpha level for the normal approximation
    Returns
    -------
    ari : float
        Adjusted Rand Index between -1.0 and 1.0. 
    ari_variance : float
        variance of the ARI
    lowerci : float 
        CI lower limit 
    upperci : float
        CI upper limit
    
    '''
    ct = contingency_matrix(clusters,trueclusters).astype(np.float)
    N = ct.sum()
    
    
    # precompute some vars
    ncomb = spcomb(N, 2)
    ctsqsum = (ct**2).sum()
    rssq = sum(ct.sum(1)**2)# row sums sq
    cssq = sum(ct.sum(0)**2)# col sums sq
    
    # use same var names as in Steinley et al
    a = (ctsqsum-N)/2.0
    b = (rssq - ctsqsum)/2.0 
    c = (cssq - ctsqsum)/2.0 # col sums sq
    d = (ctsqsum + N**2 - rssq - cssq)/2.0
    e = 2.0 * rssq - (N + 1.0) * N
    f = 2.0 * cssq - (N + 1.0) * N
    g = 4.0 * sum(ct.sum(1)**3) - 4 * (N + 1.0) * rssq + (N + 1.0)**2 * N
    h = N * (N - 1.0)
    i = 4.0 * sum(ct.sum(0)**3) - 4.0 * (N + 1.0) * cssq + (N + 1.0)**2 * N
    
    var_aplusd = 1.0/16.0 * (2.0 * N * (N - 1.0) - ((e * f) / (N * (N - 1.0)))**2 +  \
                 (4.0 * (g - h) * (i - h)) / (N * (N - 1.0) * (N - 2.0))) +  \
                 1.0/16.0 * (((e**2 - 4 * g + 2 * h) * (f**2 - 4.0 * i + 2.0 * h)) /  \
                  (N * (N - 1.0) * (N - 2.0) * (N - 3.0)))
        
    ari_variance = (ncomb**2 * var_aplusd) /((ncomb**2 - ((a + b) * (a + c) + (b + d) * (c + d)))**2)
    ari_std = np.sqrt(ari_variance)
    
    sum_comb_c = sum(spcomb(n_c, 2) for n_c in np.ravel(ct.sum(axis=1)))
    sum_comb_k = sum(spcomb(n_k, 2) for n_k in np.ravel(ct.sum(axis=0)))
    sum_comb = sum(spcomb(n_ij, 2) for n_ij in np.ravel(ct))

    prod_comb = (sum_comb_c * sum_comb_k) / ncomb
    mean_comb = (sum_comb_k + sum_comb_c) / 2.
    
    #compute ari
    ari = (sum_comb - prod_comb) / (mean_comb - prod_comb)
    #compute CI
    qnrm = norm.ppf(1.0-alpha/2.0)
    lowerci = ari - qnrm*ari_std
    upperci = ari + qnrm*ari_std
    
    return ari,ari_variance,lowerci,upperci
    
    
if __name__ == "__main__":
    clusters = np.array([0,1,1,1,1,1,1,1,1,0,0,0,0])
    trueclusters = np.array([0,1,1,1,1,0,0,1,1,1,0,0,0])
    print(ari_ci(clusters,trueclusters,alpha = 0.05))
