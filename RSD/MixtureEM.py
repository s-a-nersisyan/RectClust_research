import numpy as np
from scipy.special import logsumexp

class MixtureEM:
    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4):
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
        NLL_global_min = np.inf  # Global in terms of algorithm initializations
        for init_idx in range(self.n_init):
            arg_init = self.initialize_arg(X)
            NLL_arg_min, NLL_min = self.run_EM(X, arg_init)

            if NLL_min < NLL_global_min:
                NLL_global_arg_min = NLL_arg_min
                NLL_global_min = NLL_min

        self.NLL_global_arg_min = NLL_global_arg_min
        self.NLL_global_min = NLL_global_min
        return NLL_global_arg_min, NLL_global_min

    def predict_proba(self, X):
        '''
        Predict posterior probability of each component given the data
        '''
        return self.E_step(X, self.NLL_global_arg_min)

    def run_EM(self, X, arg_init):
        '''
        Run cycle of EM algorithm given data and initial parameters
        '''
        arg_old = arg_init
        NLL_old_min = None
        for iter_idx in range(self.max_iter):
            r = self.E_step(X, arg_old)
            arg_new, NLL_new_min = self.M_step(X, r)

            if NLL_old_min is not None and NLL_old_min < NLL_new_min:
                print(iter_idx)
                print(NLL_old_min, NLL_new_min)
                print(arg_old)
                print(arg_new)
                print("NLL is not decreasing...")
                #break

            if NLL_old_min is not None and abs(NLL_new_min - NLL_old_min) / abs(NLL_old_min) < self.tol:
                break

            arg_old = arg_new
            NLL_old_min = NLL_new_min

        # TODO: raise warning if tolerance is not achieved in max_iter iterations
        return arg_new, NLL_new_min

    def E_step(self, X, arg):
        '''
        Run E step of EM algorithm
        '''
        numerator_matrix = self.log_p_matrix(X, arg) + arg["log_pi"]
        denominator_matrix = logsumexp(numerator_matrix, axis=1)[:, None]
        r = np.exp(numerator_matrix - denominator_matrix)

        return r

    def M_step(self, X, r):
        '''
        Run M step of EM algorithm
        '''
        return self.aux_minimizer(X, r)
