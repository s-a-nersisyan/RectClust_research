import numpy as np
from scipy.special import logsumexp

from sklearn.cluster import kmeans_plusplus

from MixtureEM import MixtureEM
from RSD import RSD

from core_cython.MLE import log_p_matrix as log_p_matrix_cython
from core_cython.MLE import minimize_NLL_matrix


class RSDMixtureEM(MixtureEM):
    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4, n0=1e-6, s0=1):
        """Initialize class for RSD mixture model.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        n_init : int
            Number of initializations of EM-algorithm.
        max_iter : inr
            Maximum number of iterations for a single run of EM-algorithm.
        tol : float
            The convergence threshold for EM-algorithm.
        n0: float
            Bayesian regularizatin strength (must be positive)
        s0: float
            Desired rectangle width (for prior distribution)
        
        Returns
        -------
        None
        """
        super().__init__(n_clusters, n_init, max_iter, tol)
        self.n0 = n0
        self.s0 = s0

    def fit(self, X):
        '''
        Fit mixture model via EM algorithm given matrix X.
        This is a wrapper around MixtureEM.fit()
        '''

        # Argsort X by each feature independently
        self.X_argsort = np.argsort(X, axis=0)
        self.X_sorted = np.take_along_axis(X, self.X_argsort, axis=0)

        NLL_global_arg_min, NLL_global_min = super().fit(X)

        self.low = NLL_global_arg_min["low"]
        self.high = NLL_global_arg_min["high"]
        self.scale = NLL_global_arg_min["scale"]
        self.weights = np.exp(NLL_global_arg_min["log_pi"])

    def initialize_arg(self, X):
        '''
        Initialize EM algorithm
        '''
        n_samples, n_features = X.shape
        
        log_pi = np.log(np.full((1, self.n_clusters), 1 / self.n_clusters))
        
        low, _ = kmeans_plusplus(X, self.n_clusters)
        high = low
        scale = np.ones((self.n_clusters, n_features)) / self.n_clusters

        return {"log_pi": log_pi, "low": low, "high": high, "scale": scale}
    
    def log_p_matrix(self, X, arg):
        '''
        Compute probability matrix ( log p_k(x_i) )
        '''
        return log_p_matrix_cython(X, arg["low"], arg["high"], arg["scale"])

    def aux_minimizer(self, X, r):
        '''
        Minimize auxiliarly function given matrix r
        '''
        n_samples, n_features = X.shape
        r_sum = np.sum(r, axis=0)[None]

        # First, pi
        log_pi = np.log(r_sum / n_samples)
        low, high, scale, prior = minimize_NLL_matrix(self.X_sorted, self.X_argsort, r, self.n0, self.s0)
        
        arg_min = {"log_pi": log_pi, "low": low, "high": high, "scale": scale}
        NLL = -np.sum(logsumexp(self.log_p_matrix(X, arg_min) + arg_min["log_pi"], axis=1)) + prior
        
        return arg_min, NLL
    
    def pdf_on_component(self, x, k):
        '''
        Return density of k-th component at point x 
        after fitting the mixture model
        '''
        result = 1
        for j in range(self.low.shape[1]):
            result *= RSD.pdf(
                x[j],
                low=self.low[k, j],
                high=self.high[k, j],
                scale=self.scale[k, j]
            )

        return result
    
    def pdf(self, x):
        '''
        Return density at point x after mixture
        model is fitted
        '''
        result = 0
        for k in range(self.low.shape[0]):
            result += self.weights[0, k] * self.pdf_on_component(x, k)

        return result
