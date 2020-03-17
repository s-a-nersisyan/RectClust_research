import numpy as np

from MixtureEM import MixtureEM

class SphericalGaussianMixtureEM(MixtureEM):
    def initialize_arg(self):
        log_pi = np.full((1, self.n_clusters), 1 / self.n_clusters)
        mu = np.zeros((self.n_clusters, self.dim))  # FIXME
        alpha = np.ones((self.n_clusters, 1))  # FIXME
        return {"log_pi": log_pi, "mu": mu, "alpha": alpha}

    def log_p_matrix(self, arg):
        '''
        Compute probability matrix ( log p_k(x_i) )
        '''
        X_ext = np.repeat(self.X[:, None, :], self.n_clusters, axis=1)  # Extend X by adding new axis
        log_p_1 = self.dim * (np.log(2) - np.sqrt(np.pi))
        log_p_2 = self.dim * np.log(2 * arg["alpha"]).T
        log_p_3 = -np.sum(((X_ext - arg["mu"]) / arg["alpha"])**2, axis=2)

        return log_p_1 + log_p_2 + log_p_3


    def NLL_minimizer(self, r):
        return 17, 17  # FIXME
