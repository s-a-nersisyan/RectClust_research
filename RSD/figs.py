import matplotlib.pyplot as plt
import numpy as np

from RSD import RSD

l, h = -6, 4
xs = np.linspace(l - 4, h + 4, 1000)
ys = np.vectorize(RSD.pdf)(xs, low=l, high=h)

plt.plot(xs, ys, c="black")

plt.xticks(list(range(l - 4, h + 5, 2)))
plt.savefig("RSD_normal.pdf")
plt.tight_layout()
plt.close()

l, h = -6, 4
k = lambda x: np.maximum(0, x - h) - np.maximum(0, l - x)
xs = np.linspace(l - 4, h + 4, 1000)
ys = k(xs)
plt.plot(xs, ys, c="black")

plt.axis("equal")
plt.xticks(list(range(l - 4, h + 5, 2)))
plt.savefig("k.pdf")
plt.tight_layout()
plt.close()

xs = [-6, -1, 0, 2, 3]
x1, x2 = min(xs), max(xs)

plt.plot([x1, x2], [x1, x2], c="black")

for x in xs:
    plt.plot([x, x], [x, x2], "--", c="black")
    plt.plot([x1, x], [x, x], "--", c="black")

plt.axis("equal")
plt.xticks(list(range(x1 - 1, x2 + 2)))
plt.yticks(list(range(x1 - 1, x2 + 2)))

plt.tight_layout()
plt.savefig("regions.pdf")
