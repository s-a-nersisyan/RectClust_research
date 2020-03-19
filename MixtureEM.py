import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

class MixtureEM:
    def __init__(self, n_clusters=2, n_init=1, max_iter=300, tol=1e-4):
        '''
        Abstract class implementing EM algorithm for
        the problem of mixture separation.
        '''
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        '''
        Fit mixture model via EM algorithm given matrix X
        '''
        # TODO: scaling?
        self.X = X
        self.n_samples, self.dim = self.X.shape
        # Extend X by new axis to fasten computations
        self.X_ext = np.repeat(self.X[:, None, :], self.n_clusters, axis=1)

        # TODO: select the best
        for init_idx in range(self.n_init):
            # TODO: generate initial parameters
            arg_init = self.initialize_arg()
            NLL_arg_min, NLL_min = self.run_EM(arg_init)

    def run_EM(self, arg_init):
        arg_old = arg_init
        NLL_old_min = None
        for iter_idx in range(self.max_iter):
            r = self.E_step(arg_old)
            arg_new, NLL_new_min = self.M_step(r)

            #fig, ax = plt.subplots()
            #ax.plot(self.X[:, 0], self.X[:, 1], "o")
            #ax.plot(arg_new["mu"][:, 0], arg_new["mu"][:, 1], "x")
            #for mu, alpha in zip(arg_new["mu"], arg_new["alpha"]):
            #    ax.add_artist(plt.Circle(mu, alpha*3/np.sqrt(2), fill=0))
            ##plt.show()
            #plt.close()

            if not NLL_old_min is None and abs(NLL_new_min - NLL_old_min) / abs(NLL_old_min) < self.tol:
                break

            arg_old = arg_new
            NLL_old_min = NLL_new_min

        # TODO: raise warning if tolerance is not achieved in max_iter iterations
        return arg_new, NLL_new_min

    def E_step(self, arg_old):
        numerator_matrix = self.log_p_matrix(arg_old) + arg_old["log_pi"]
        denominator_matrix = logsumexp(numerator_matrix, axis=1)[:, None]
        r = np.exp(numerator_matrix - denominator_matrix)
        # TODO: test that this is true probability matrix (rows sum to 1)
        return r

    def M_step(self, r):
        return self.aux_minimizer(r)
