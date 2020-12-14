import numpy as np

from RSD import RSD

from scipy.stats import norm, shapiro

import matplotlib.pyplot as plt
import seaborn as sns


# Sample parameters
low, high, scale = -5, 5, 1
N = 1000000

C = 1 / (np.sqrt(2 * np.pi) * scale + high - low)

E_ll = C * (np.sqrt(np.pi / 2) / scale - C)
E_lh = C**2
E_ls = C * (np.sqrt(2 * np.pi) * C - 2 / scale)
E_hh = C * (np.sqrt(np.pi / 2) / scale - C)
E_hs = - C * (np.sqrt(2 * np.pi) * C - 2 / scale)
E_ss = C * ( 3 * np.sqrt(2 * np.pi) / scale - 2 * np.pi * C)

inf_matrix = np.array([
    [E_ll, E_lh, E_ls],
    [E_lh, E_hh, E_hs],
    [E_ls, E_hs, E_ss],
])
asympt_var_matrix = np.linalg.inv(inf_matrix)
print(asympt_var_matrix)

err_l, err_h, err_s = [], [], []
for i in range(1000):
    sample = RSD.rvs(low=low, high=high, scale=scale, size=N, random_state=i**3)
    low_MLE, high_MLE, scale_MLE, NLL_min = RSD.fit(sample)
    err_l.append(np.sqrt(N) * (low_MLE - low))
    err_h.append(np.sqrt(N) * (high_MLE - high))
    err_s.append(np.sqrt(N) * (scale_MLE - scale))

err_l = np.array(err_l)
err_h = np.array(err_h)
err_s = np.array(err_s)

sns.histplot(err_l, kde=True)
plt.savefig("distr_low.pdf")
plt.close()
sns.histplot(err_h, kde=True)
plt.savefig("distr_high.pdf")
plt.close()
sns.histplot(err_s, kde=True)
plt.savefig("distr_scale.pdf")
plt.close()

print(np.mean(err_l), np.var(err_l), np.mean(err_l**2))
print(np.mean(err_h), np.var(err_h), np.mean(err_h**2))
print(np.mean(err_s), np.var(err_s), np.mean(err_s**2))
