import numpy as np
from scipy.special import logsumexp, gamma
from scipy.optimize import *

from MixtureEM import MixtureEM


class GeneralizedGaussianMixtureEM(MixtureEM):
    def __init__(self, beta, **kwargs):
        # TODO: explicit arguments
        super().__init__(**kwargs)
        self.beta = beta

    def fit(self, X):
        '''
        Fit mixture model via EM algorithm given matrix X.
        This is a wrapper around MixtureEM.fit()
        '''
        NLL_global_arg_min, NLL_global_min = super().fit(X)

        self.centers = NLL_global_arg_min["mu"]
        self.edge_lengths = 2 * NLL_global_arg_min["alpha"]
        self.weights = np.exp(NLL_global_arg_min["log_pi"])

    def initialize_arg(self, X):
        '''
        Initialize EM algorithm
        '''
        n_samples, n_features = X.shape

        log_pi = np.log(np.full((1, self.n_clusters), 1 / self.n_clusters))
        mu = np.random.sample((self.n_clusters, n_features))
        alpha = np.ones((self.n_clusters, n_features))

        return {"log_pi": log_pi, "mu": mu, "alpha": alpha}

    def log_p_matrix(self, X, arg):
        '''
        Compute probability matrix ( log p_k(x_i) )
        '''
        n_samples, n_features = X.shape
        X_ext = np.repeat(X[:, None, :], self.n_clusters, axis=1)

        log_p_1 = n_features * np.log(self.beta / (2 * gamma(1 / self.beta)))
        log_p_2 = -np.sum(np.log(arg["alpha"]), axis=1)[None]
        log_p_3 = -np.sum((np.abs(X_ext - arg["mu"]) / arg["alpha"])**self.beta, axis=2)
        return log_p_1 + log_p_2 + log_p_3

    def aux_minimizer(self, X, r):
        '''
        Minimize auxiliarly function given matrix r
        '''
        n_samples, n_features = X.shape
        # TODO: measure time of this line
        X_ext = np.repeat(X[:, None, :], self.n_clusters, axis=1)
        r_sum = np.sum(r, axis=0)[None]

        # First, pi
        log_pi = np.log(r_sum / n_samples)

        # Then, mu
        # TODO: vectorize me
        mu = np.zeros((self.n_clusters, n_features))
        for k in range(self.n_clusters):
            for j in range(n_features):
                mu[k, j] = minimize_scalar(lambda t: self.m(t, k, j, X, r)).x

        # Perform several Newton iterations
        #iternum = 1
        #for k in range(self.n_clusters):
        #    for j in range(n_features):
        #mu

        # Finally, alpha
        alpha_numerator = np.einsum("ik,ikj->kj", r, np.abs(X_ext - mu)**self.beta)
        alpha_denominator = r_sum.T
        alpha = (self.beta * alpha_numerator / r_sum.T) ** (1 / self.beta)

        arg_min = {"log_pi": log_pi, "mu": mu, "alpha": alpha}

        # Now compute NLL at the argument found
        NLL = -np.sum(logsumexp(self.log_p_matrix(X, arg_min) + arg_min["log_pi"], axis=1))

        return arg_min, NLL

    def m(self, t, k, j, X, r, derivative_order=0):
        '''
        Compute technical function m_k^j(t)
        '''
        d = derivative_order
        mul_factor = np.prod(range(self.beta - d + 1, self.beta + 1))
        return mul_factor * np.dot(r[:, k], np.sign(X[:, j] - t)**d * np.abs(X[:, j] - t)**(self.beta - d))
