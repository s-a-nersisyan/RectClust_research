import numpy as np


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
        # TODO: scaling?
        self.X = X
        self.n_samples, self.dim = self.X.shape

        for init_idx in range(n_init):
            # TODO: generate initial parameters
            l_arg_min, l_min = self.run_EM()

    def run_EM(self):
        l_prev_min = None
        for iter_idx in range(max_iter):
            r = self.E_step()
            l_cur_arg_min, l_cur_min = self.M_step(r)

            if l_prev_min is None:
                l_prev_min = l_cur_min
            else:
                if abs(l_cur_min - l_prev_min) / abs(l_prev_min) < self.tol:
                    break
        # TODO: raise warning if tolerance is not achieved in max_iter iterations
        return l_cur_arg_min, l_cur_min

    def E_step(self):
        pass

    def M_step(self):
        pass
