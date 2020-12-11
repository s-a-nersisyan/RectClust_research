import numpy as np

from RSD import RSD

from scipy.stats import linregress

import matplotlib.pyplot as plt
import seaborn as sns


# Sample parameters
low, high, scale = 0, 5, 0

# First, test that MLE error decreases with sqrt(N) rate
low_err, high_err, scale_err = [], [], []
Ns = np.logspace(10, 20, 1000, base=2)
for N in Ns:
    sample = RSD.rvs(low=low, high=high, scale=scale, size=int(N), random_state=int(N))
    low_MLE, high_MLE, scale_MLE, NNL_min = RSD.fit(sample)
    low_err.append(abs(low_MLE - low))
    high_err.append(abs(high_MLE - high))
    scale_err.append(abs(scale_MLE - scale))

low_err = np.array(low_err)
high_err = np.array(high_err)
scale_err = np.array(scale_err)

sns.scatterplot(x=np.log2(Ns), y=np.log2(low_err))
plt.savefig("N_vs_low_err_{}_{}_{}.pdf".format(low, high, scale))
plt.close()
sns.scatterplot(x=np.log2(Ns), y=np.log2(high_err))
plt.savefig("N_vs_high_err_{}_{}_{}.pdf".format(low, high, scale))
plt.close()
sns.scatterplot(x=np.log2(Ns), y=np.log2(scale_err))
plt.savefig("N_vs_scale_err_{}_{}_{}.pdf".format(low, high, scale))
plt.close()

print("N vs low_err: ", linregress(np.log2(Ns), np.log2(low_err)))
print("N vs high_err: ", linregress(np.log2(Ns), np.log2(high_err)))
print("N vs scale_err: ", linregress(np.log2(Ns), np.log2(scale_err)))
