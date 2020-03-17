import numpy as np

from GMM import SphericalGaussianMixtureEM

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
model = SphericalGaussianMixtureEM()
model.fit(X)
