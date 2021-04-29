import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from RSD import RSD
from RSD_mixture import RSDMixtureEM

import time

np.random.seed(17)

# Synthetic test
N = 10000
clusters = [
    ((-4, 0, 0.5), (0, 2, 0.1)),
    ((4, 5, 0), (-2, 0, 0.3)),
    ((-3, -2, 0.1), (-1, 1, 0.2))
]

Xs = []
for i, (x, y) in enumerate(clusters):
    X = RSD.rvs(low=x[0], high=x[1], scale=x[2], size=N)[:, None]
    Y = RSD.rvs(low=y[0], high=y[1], scale=y[2], size=N)[:, None]
    Xs.append(np.concatenate([X, Y], axis=1))

X = np.concatenate(Xs, axis=0)
df = pd.DataFrame(X, columns=["x", "y"])
df["Cluster"] = ["Cluster #1"]*N + ["Cluster #2"]*N + ["Cluster #3"] * N
ax = sns.scatterplot(x="x", y="y", hue="Cluster", data=df)
plt.xlabel("")
plt.ylabel("")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])

model = RSDMixtureEM(n_clusters=len(clusters), n_init=100)

start = time.time()
model.fit(X)
end = time.time()
print("{:.1f}".format(end - start))

for k in [2, 1, 0]:
    print("[{:.1f}, {:.1f}] x [{:.1f}, {:.1f}] ({}, {})".format(
        model.low[k, 0], model.high[k, 0], model.low[k, 1], model.high[k, 1], model.scale[k, 0], model.scale[k, 1])
    )
    plt.plot(
        [model.low[k, 0], model.low[k, 0], model.high[k, 0], model.high[k, 0], model.low[k, 0]], 
        [model.low[k, 1], model.high[k, 1], model.high[k, 1], model.low[k, 1], model.low[k, 1]],
        linewidth=4
    )
plt.tight_layout()
plt.savefig("synthetic.png", dpi=300)
