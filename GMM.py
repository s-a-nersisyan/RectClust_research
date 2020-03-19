import numpy as np
from scipy.special import logsumexp

from MixtureEM import MixtureEM

class SphericalGaussianMixtureEM(MixtureEM):
    def initialize_arg(self):
        log_pi = np.log(np.full((1, self.n_clusters), 1 / self.n_clusters))
        mu = np.zeros((self.n_clusters, self.dim))  # FIXME
        alpha = np.ones((self.n_clusters, 1))  # FIXME
        return {"log_pi": log_pi, "mu": mu, "alpha": alpha}

    def log_p_matrix(self, arg):
        '''
        Compute probability matrix ( log p_k(x_i) )
        '''
        log_p_1 = self.dim * (np.log(2) - np.sqrt(np.pi))
        log_p_2 = -self.dim * np.log(2 * arg["alpha"]).T
        log_p_3 = -np.sum(((self.X_ext - arg["mu"]) / arg["alpha"])**2, axis=2)


        return log_p_1 + log_p_2 + log_p_3


    def aux_minimizer(self, r):
        '''
        Minimize auxiliarly function given matrix r
        '''
        #r = np.ones((self.n_samples, self.n_clusters))

        # First, pi
        log_pi = np.log(np.sum(r, axis=0)[None] / self.n_samples)
        # Then, mu
        mu_numerator = np.einsum("ik,ij->kj", r, self.X)
        mu_denominator = np.sum(r, axis=0)[None].T
        mu = mu_numerator / mu_denominator
        # Finally, alpha
        alpha = (2 * np.sum(np.einsum("ik,ikj->kj", r, (self.X_ext - mu)**2), axis=1) / (self.n_samples * self.dim)) ** 0.5
        arg_min = {"log_pi": log_pi, "mu": mu, "alpha": alpha[:, None]}

        # Now compute NLL at the argument found
        NLL = -np.sum(logsumexp(self.log_p_matrix(arg_min) + arg_min["log_pi"], axis=1))
        return arg_min, NLL
