from RSD import RSD
from mle_pyx import traverse_extremal_chain as tec_pyx

import numpy as np
import time

low, high, scale = -4, 3, 0.1
data = RSD.rvs(low=low, high=high, scale=scale, size=10000000, random_state=0)
data = np.sort(data)
weights = np.ones(len(data), dtype=np.float64)

start = time.time()
#res1 = RSD.fit(data, weights)
res1 = np.mean(data), np.std(data)
end = time.time()
t1 = end - start

start = time.time()
res2 = tec_pyx(data, weights)
end = time.time()
t2 = end - start
print(res1)
print(res2)

print(t1, t2, t1 / t2)
