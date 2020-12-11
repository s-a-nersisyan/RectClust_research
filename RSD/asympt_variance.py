import numpy as np

from RSD import RSD

from scipy.stats import norm, shapiro

import matplotlib.pyplot as plt
import seaborn as sns


# Sample parameters
low, high, scale = 0, 0, 17
N = 1000000

print(np.mean(np.array([np.sqrt(N) * (norm.fit(norm.rvs(loc=low, scale=scale, size=N))[1] - scale) for i in range(100)])**2))
print(scale ** 2 / 2)
quit()
#print(np.var([np.sqrt(N) * (norm.fit(RSD.rvs(low=low, high=high, scale=scale, size=N))[1] - scale) for i in range(100)]) / scale**2)

for i in range(100):
    sample = norm.rvs(loc=low, scale=scale, size=N)
    low_MLE, high_MLE, scale_MLE, NLL_min = RSD.fit(sample)
    
    print(RSD.NLL(sample, low, high, scale), RSD.NLL(sample, low_MLE, high_MLE, scale_MLE))
    
    #print(NLL_min)
    #print(RSD.NLL(sample, low, high, scale))
    #print(high_MLE - low_MLE)

quit()

print(np.var([np.sqrt(N) * (RSD.fit(norm.rvs(loc=low, scale=scale, size=N))[2] - scale) for i in range(100)]) / scale**2)

print(np.var([np.sqrt(N) * (RSD.fit(RSD.rvs(low=low, high=high, scale=scale, size=N))[2] - scale) for i in range(100)]) / scale**2)
print(np.var([np.sqrt(N) * (RSD.fit(norm.rvs(loc=low, scale=scale, size=N))[2] - scale) for i in range(100)]) / scale**2)

print(scale ** 2 / 2)


quit()
# First, test that MLE error decreases with sqrt(N) rate
low_err, high_err, scale_err = [], [], []
Ns = np.logspace(10, 20, 1000, base=2)
for N in Ns:
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
