import numpy as np
from scipy.special import logsumexp

from MixtureEM import MixtureEM
from RSD import RSD
from core_cython.EM_utils import log_p_matrix as log_p_matrix_cython


class RSDMixtureEM(MixtureEM):
    def fit(self, X):
        '''
        Fit mixture model via EM algorithm given matrix X.
        This is a wrapper around MixtureEM.fit()
        '''
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
        #np.random.seed(3)
        low = np.random.sample((self.n_clusters, n_features))
        high = low
        scale = np.ones((self.n_clusters, n_features))

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
        
        low = np.empty((self.n_clusters, n_features))
        high = np.empty((self.n_clusters, n_features))
        scale = np.empty((self.n_clusters, n_features))
        
        # TODO: cythonize me, please
        for k in range(self.n_clusters):
            for j in range(n_features):
                low_MLE, high_MLE, scale_MLE, NLL_min = RSD.fit(X[:, j], weights=r[:, k])
                low[k, j] = low_MLE
                high[k, j] = high_MLE
                scale[k, j] = scale_MLE
                
                #if scale_MLE < 0:
                #    print(k, j)
                #    print(low_MLE, high_MLE, scale_MLE)
                #    print(NLL_min)
                #    print(RSD.NLL(X[:, j], low=low_MLE, high=high_MLE, scale=scale_MLE, weights=r[:, k]))
                #
                #if k == 0 and j == 0:
                #    print("***")
                #    print(low_MLE, high_MLE, scale_MLE)
                #    print(NLL_min)
                #    #low_MLE, high_MLE, scale_MLE = -8.143519790713134, 5.090250373797931, 0.04477668846863345
                #    print(RSD.NLL(X[:, j], low=low_MLE, high=high_MLE, scale=scale_MLE, weights=r[:, k]))
                #    print("***")
        
        arg_min = {"log_pi": log_pi, "low": low, "high": high, "scale": scale}
        NLL = -np.sum(logsumexp(self.log_p_matrix(X, arg_min) + arg_min["log_pi"], axis=1))

        return arg_min, NLL


if __name__ == "__main__":
    rs = 17
    X1 = RSD.rvs(low=-4, high=0, scale=0.1, size=500, random_state=rs)[:, None]
    Y1 = RSD.rvs(low=0, high=2, scale=0.1, size=500, random_state=rs + 1)[:, None]
    S1 = np.concatenate([X1, Y1], axis=1)
    
    X2 = RSD.rvs(low=4, high=5, scale=0.1, size=1000, random_state=rs + 2)[:, None]
    Y2 = RSD.rvs(low=-2, high=0, scale=0.1, size=1000, random_state=rs + 3)[:, None]
    S2 = np.concatenate([X2, Y2], axis=1)
    
    X3 = RSD.rvs(low=-3, high=-2, scale=0.1, size=1500, random_state=rs + 4)[:, None]
    Y3 = RSD.rvs(low=-1, high=1, scale=0.1, size=1500, random_state=rs + 5)[:, None]
    S3 = np.concatenate([X3, Y3], axis=1)

    X = np.concatenate([S1, S2, S3], axis=0)

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.DataFrame(X, columns=["x", "y"])
    df["Cluster"] = ["0"]*500 + ["1"]*1000 + ["2"] * 1500
    sns.scatterplot(x="x", y="y", hue="Cluster", data=df)

    model = RSDMixtureEM(n_clusters=3, n_init=100)
    model.fit(X)

    for k in range(3):
        print("[{:.1f}, {:.1f}] x [{:.1f}, {:.1f}] ({}, {})".format(
            model.low[k, 0], model.high[k, 0], model.low[k, 1], model.high[k, 1], model.scale[k, 0], model.scale[k, 1])
        )
        plt.plot(
            [model.low[k, 0], model.low[k, 0], model.high[k, 0], model.high[k, 0], model.low[k, 0]], 
            [model.low[k, 1], model.high[k, 1], model.high[k, 1], model.low[k, 1], model.low[k, 1]]
        )
    plt.tight_layout()
    plt.savefig("test.pdf")
