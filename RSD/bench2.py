import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from RSD import RSD
from scipy.stats import norm

'''
locs = []
scales = []
for i in range(100):
    sample = norm.rvs(size=1000, random_state=i)
    loc, scale = norm.fit(sample)
    locs.append(loc)
    scales.append(scale)

print("Normal")
print(np.quantile(locs, 0.75) - np.quantile(locs, 0.25))
print(np.quantile(scales, 0.75) - np.quantile(scales, 0.25))


lows = []
highs = []
scales = []
for i in range(100):
    sample = RSD.rvs(low=-3, high=40, scale=1, size=100, random_state=i)
    low, high, scale, _ = RSD.fit(sample)
    lows.append(low)
    highs.append(high)
    scales.append(scale)

print("RSD")
print(np.quantile(lows, 0.75) - np.quantile(lows, 0.25))
print(np.quantile(highs, 0.75) - np.quantile(highs, 0.25))
print(np.quantile(scales, 0.75) - np.quantile(scales, 0.25))
'''


low, high, scale = -3, 3, 0.1
sample = RSD.rvs(low=low, high=high, scale=scale, size=1000, random_state=0)

x_grid = np.linspace(low - 5*scale - 1, high + 5*scale + 1, 1000)
pdf_grid = np.array([RSD.pdf(x, low, high, scale) for x in x_grid])

plt.plot(x_grid, pdf_grid)
sns.histplot(sample, stat="density")
plt.show()


quit()
low_MLE, high_MLE, scale_MLE, _ = RSD.fit(sample)
#low_MLE, high_MLE, scale_MLE = -1, 2, 2.5
print(low_MLE, high_MLE, scale_MLE,)
quit()
#print(low, high, scale)
#print(low_MLE, high_MLE, scale_MLE)
print(RSD.NLL(sample, low, high, scale))
print(RSD.NLL(sample, low_MLE, high_MLE, scale_MLE))


#quit()
#sample = RSD.rvs(low=low_MLE, high=high_MLE, scale=scale_MLE, size=10000, random_state=0)
#low_MLE2, high_MLE2, scale_MLE2, _ = RSD.fit(sample)
#print(low_MLE, high_MLE, scale_MLE)
#print(low_MLE2, high_MLE2, scale_MLE2)
##print(RSD.NLL(sample, low, high, scale))
##print(RSD.NLL(sample, low_MLE, high_MLE, scale_MLE))
#
##sns.histplot(sample)
##plt.show()
#
#quit()

xs = np.linspace(low - 5*scale - 1, high + 5*scale + 1, 1000)
ys = np.array(list(map(lambda x: RSD.pdf(x, low, high, scale), xs)))
plt.plot(xs, ys)
ys = np.array(list(map(lambda x: RSD.pdf(x, low, high, scale), sample)))
plt.plot(sample, ys, "x")
plt.tight_layout()
plt.show()
