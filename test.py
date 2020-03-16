from GMM import GaussianMixtureEM
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
model = GaussianMixtureEM()
model.fit(X)
