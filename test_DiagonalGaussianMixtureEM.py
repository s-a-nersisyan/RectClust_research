import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

from GMM import DiagonalGaussianMixtureEM

#np.random.seed(1)

X1 = np.random.normal(loc=-3, scale=1, size=50)[:, None]
Y1 = np.random.normal(loc=-1, scale=2, size=50)[:, None]
S1 = np.concatenate([X1, Y1], axis=1)

X2 = np.random.normal(loc=2, scale=3.7, size=100)[:, None]
Y2 = np.random.normal(loc=4, scale=7.7, size=100)[:, None]
S2 = np.concatenate([X2, Y2], axis=1)

X3 = np.random.normal(loc=0, scale=3, size=350)[:, None]
Y3 = np.random.normal(loc=-5, scale=0.2, size=350)[:, None]
S3 = np.concatenate([X3, Y3], axis=1)

X = np.concatenate([S1, S2, S3], axis=0)
#plt.plot(X[:, 0], X[:, 1], "o")
#plt.show()

#model = GaussianMixture(n_components=2, covariance_type="spherical", tol=1e-4, reg_covar=0, verbose=2, verbose_interval=1, init_params="random")
#model.fit(X)
#
#print(model.means_)
#quit()

model = DiagonalGaussianMixtureEM(n_clusters=3)
model.fit(X)
r = model.predict_proba(X)
base_colors = np.repeat(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[None], len(X), axis=0)
colors = np.einsum("ik,ikd->id", r, base_colors)
print(model.means)
print(model.SDs)
print(model.weights)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=colors)

ax.plot(model.means[:, 0], model.means[:, 1], "x")
for mean, SD in zip(model.means, model.SDs):
    ax.add_artist(Ellipse(mean, 6*SD[0], 6*SD[1], fill=0))

ax.axis('equal')
plt.show()
plt.close()
