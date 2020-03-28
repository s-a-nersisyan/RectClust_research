import numpy as np
from scipy.special import logsumexp

from MixtureEM import MixtureEM


class SphericalGaussianMixtureEM(MixtureEM):
    def fit(self, X):
        '''
        Fit mixture model via EM algorithm given matrix X.
        This is a wrapper around MixtureEM.fit()
        '''
        NLL_global_arg_min, NLL_global_min = super().fit(X)

        self.means = NLL_global_arg_min["mu"]
        self.SDs = NLL_global_arg_min["alpha"] / np.sqrt(2)
        self.weights = np.exp(NLL_global_arg_min["log_pi"])

    def initialize_arg(self, X):
        '''
        Initialize EM algorithm
        '''
        n_samples, n_features = X.shape

        log_pi = np.log(np.full((1, self.n_clusters), 1 / self.n_clusters))
        mu = np.random.sample((self.n_clusters, n_features))
        alpha = np.ones((self.n_clusters, 1))

        return {"log_pi": log_pi, "mu": mu, "alpha": alpha}

    def log_p_matrix(self, X, arg):
        '''
        Compute probability matrix ( log p_k(x_i) )
        '''
        n_samples, n_features = X.shape
        X_ext = np.repeat(X[:, None, :], self.n_clusters, axis=1)

        log_p_1 = n_features * np.log(np.sqrt(np.pi))
        log_p_2 = -n_features * np.log(arg["alpha"]).T
        log_p_3 = -np.sum(((X_ext - arg["mu"]) / arg["alpha"])**2, axis=2)
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
        mu = np.einsum("ik,ij->kj", r, X) / r_sum.T

        # Finally, alpha
        alpha_numerator = np.einsum("ik,ikj->kj", r, (X_ext - mu)**2)
        alpha_denominator = r_sum.T
        alpha = (2 * np.sum(alpha_numerator / alpha_denominator, axis=1) / n_features) ** 0.5

        arg_min = {"log_pi": log_pi, "mu": mu, "alpha": alpha[:, None]}

        # Now compute NLL at the argument found
        NLL = -np.sum(logsumexp(self.log_p_matrix(X, arg_min) + arg_min["log_pi"], axis=1))

        return arg_min, NLL


class DiagonalGaussianMixtureEM(MixtureEM):
    def fit(self, X):
        '''
        Fit mixture model via EM algorithm given matrix X.
        This is a wrapper around MixtureEM.fit()
        '''
        NLL_global_arg_min, NLL_global_min = super().fit(X)

        self.means = NLL_global_arg_min["mu"]
        self.SDs = NLL_global_arg_min["alpha"] / np.sqrt(2)
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

        log_p_1 = n_features * np.log(np.sqrt(np.pi))
        log_p_2 = -np.sum(np.log(arg["alpha"]), axis=1)[None]
        log_p_3 = -np.sum(((X_ext - arg["mu"]) / arg["alpha"])**2, axis=2)
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
        mu = np.einsum("ik,ij->kj", r, X) / r_sum.T

        # Finally, alpha
        alpha_numerator = np.einsum("ik,ikj->kj", r, (X_ext - mu)**2)
        alpha_denominator = r_sum.T
        alpha = (2 * alpha_numerator / r_sum.T) ** 0.5

        arg_min = {"log_pi": log_pi, "mu": mu, "alpha": alpha}

        # Now compute NLL at the argument found
        NLL = -np.sum(logsumexp(self.log_p_matrix(X, arg_min) + arg_min["log_pi"], axis=1))

        return arg_min, NLL
