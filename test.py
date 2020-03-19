import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from GMM import SphericalGaussianMixtureEM

np.random.seed(1)

X1 = np.random.normal(loc=-3, scale=1, size=50)[:, None]
Y1 = np.random.normal(loc=-1, scale=1, size=50)[:, None]
S1 = np.concatenate([X1, Y1], axis=1)

X2 = np.random.normal(loc=2, scale=1.7, size=100)[:, None]
Y2 = np.random.normal(loc=4, scale=1.7, size=100)[:, None]
S2 = np.concatenate([X2, Y2], axis=1)

X3 = np.random.normal(loc=0, scale=0.5, size=150)[:, None]
Y3 = np.random.normal(loc=-5, scale=0.5, size=150)[:, None]
S3 = np.concatenate([X3, Y3], axis=1)

X = np.concatenate([S1, S2], axis=0)
#plt.plot(X[:, 0], X[:, 1], "o")
#plt.show()

#model = GaussianMixture(n_components=2, covariance_type="spherical", tol=1e-4, reg_covar=0, verbose=2, verbose_interval=1, init_params="random")
#model.fit(X)
#
#print(model.means_)
#quit()

model = SphericalGaussianMixtureEM()
model.fit(X)
