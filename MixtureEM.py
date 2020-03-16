import numpy as np


class MixtureEM:
    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4):
        '''
        Abstract class implementing EM algorithm for
        the problem of mixture separation.
        '''

    def fit(self, X)
        for init_idx in range(n_init):
            # TODO: generate initial parameters
            l_arg_min, l_min = self.run_EM()

    def run_EM(self):
        pass

    def E_step(self):
        pass

    def M_step(self):
        pass
