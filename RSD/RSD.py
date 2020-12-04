import numpy as np
import numpy.polynomial.polynomial as poly
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


class RSD:
    @staticmethod
    def cdf(x, low=-1, high=1, scale=1):
        # C is a weight under Gaussian part
        C = 1 / (1 + (high - low) / np.sqrt(2*np.pi) / scale)

        if x <= low:
            return C * stats.norm.cdf(x, low, scale)
        elif low < x <= high:
            return C/2 + (1 - C) * stats.uniform.cdf(x, low, high - low)
        else:
            return 1 - C/2 + C * (stats.norm.cdf(x, high, scale) - 1/2)

    @staticmethod
    def pdf(x, low=-1, high=1, scale=1):

        if np.isclose(scale, 0):
            if low <= x <= high:
                return 1 / (high - low)
            else:
                return 0

        # C is a weight under Gaussian part
        C = 1 / (1 + (high - low) / np.sqrt(2*np.pi) / scale)

        if x <= low:
            return C * stats.norm.pdf(x, low, scale)
        elif low < x <= high:
            return C / (np.sqrt(2*np.pi) * scale)
        else:
            return C * stats.norm.pdf(x, high, scale)

    @staticmethod
    def rvs(low=-1, high=1, scale=1, size=1, random_state=None):
        # C is a weight under Gaussian part
        C = 1 / (1 + (high - low) / np.sqrt(2*np.pi) / scale)

        sample = np.zeros(size)
        # Ones correspond to Gaussian part, zeros to uniform part
        loc = stats.bernoulli.rvs(C, size=size, random_state=random_state)
        sample[loc == 0] = stats.uniform.rvs(low, high - low, size=np.sum(loc == 0), random_state=random_state)
        sample[loc == 1] = stats.norm.rvs(0, scale, size=np.sum(loc == 1), random_state=random_state)
        sample[(loc == 1) & (sample >= 0)] += high
        sample[(loc == 1) & (sample < 0)] += low

        return sample

    @staticmethod
    def fit(data, weights=None):
        '''
        Perform MLE estimation
        '''
        N = len(data)
        if not weights:
            weights = np.ones(N)
        r_sum = np.sum(weights)

        data = np.sort(data)

        u, v = 0, N - 1
        l_cur, h_cur = data[0], data[-1]
        r_l, r_h = weights[0], weights[-1]
        x_l, x_h = weights[0] * data[0], weights[-1] * data[-1]
        x_sq_l, x_sq_h = weights[0] * data[0]**2, weights[-1] * data[-1]**2
        l_min, h_min, s_min, NLL_min = data[0], data[-1], 0, np.inf

        while v - u >= 1:
            if v - u > 1:
                l_next = min(data[u + 1], (x_l + x_h - r_h * data[v - 1]) / r_l)
                h_next = max(data[v - 1], (x_l + x_h - r_l * data[u + 1]) / r_h)
            else:
                l_next = (x_l + x_h) / (r_l + r_h)
                h_next = l_next

            l_loc_min, s_loc_min, NLL_loc_min = RSD._minimize_g(
                r_sum, r_l, r_h,
                x_l, x_h,
                x_sq_l, x_sq_h,
                l_cur, l_next
            )
            h_loc_min = (x_l + x_h - r_l * l_loc_min) / r_h

            #print(l_cur, h_cur, RSD.NLL(data, low=l_loc_min, high=h_loc_min, scale=s_loc_min))
            #print(NLL_loc_min)
            #print("*******************************************************")

            if NLL_loc_min <= NLL_min:
                l_min, h_min, s_min, NLL_min = l_loc_min, h_loc_min, s_loc_min, NLL_loc_min

            if l_next == data[u + 1]:
                r_l += weights[u + 1]
                x_l += weights[u + 1] * data[u + 1]
                x_sq_l += weights[u + 1] * data[u + 1]**2
                u += 1
            else:
                r_h += weights[v - 1]
                x_h += weights[v - 1] * data[v - 1]
                x_sq_h += weights[v - 1] * data[v - 1]**2
                v -= 1

            l_cur, h_cur = l_next, h_next

        return l_min, h_min, s_min, NLL_min

    @staticmethod
    def NLL(data, low=-1, high=1, scale=1, weights=None):
        N = len(data)
        if not weights:
            weights = np.ones(N)
        r_sum = np.sum(weights)

        if np.isclose(scale, 0):
            return r_sum * np.log(high - low)
        else:
            result = r_sum * np.log(np.sqrt(2*np.pi) * scale + high - low)
            result += np.sum(weights * (np.maximum(0, data - high) - np.maximum(0, low - data))**2) / (2 * scale**2)
            return result

    @staticmethod
    def _g(t, s, r_sum, a, b, c):
        if np.isclose(s, 0):
            return r_sum * np.log(-a*t + b)
        else:
            return r_sum * np.log(np.sqrt(2*np.pi) * s - a*t + b) + (a * t**2 + c) / (2 * s**2)

    @staticmethod
    def _minimize_g(r_sum, r_l, r_h, x_l, x_h, x_sq_l, x_sq_h, l1, l2):
        a = 1 / r_l + 1 / r_h
        b = x_h / r_h - x_l / r_l
        c = x_sq_l + x_sq_h - x_l**2 / r_l - x_h**2 / r_h

        t1 = r_l * l1 - x_l
        t2 = r_l * l2 - x_l

        P = np.array([
            r_sum * c**2,
            0,
            2 * c * (r_sum * a - np.pi),
            -2 * np.pi * b,
            r_sum * a**2
        ])

        roots_t = poly.polyroots(P)
        critical_t = [t.real for t in roots_t if np.isclose(t.imag, 0)]
        critical_t = [t for t in critical_t if t1 < t < t2]
        critical_t += [t1, t2]
        critical_s = [(a * t**2 + c) / (np.sqrt(2*np.pi) * t) if not np.isclose(t, 0) else 0 for t in critical_t]

        critical_g = [RSD._g(t, s, r_sum, a, b, c) for t, s in zip(critical_t, critical_s)]

        i = np.argmin(critical_g)
        t_loc_min, s_loc_min = critical_t[i], critical_s[i]
        l_loc_min = (t_loc_min + x_l) / r_l

        return l_loc_min, s_loc_min, critical_g[i]


if __name__ == "__main__":
    #l, h, s = -1, 1, 1
    #N = 1000

    #ls = []
    #hs = []
    #ss = []
    #for i in range(100):
    #    print(i)
    #    sample = RSD.rvs(low=l, high=h, scale=s, size=N, random_state=i)
    #    l_mle, h_mle, s_mle, _ = RSD.fit(sample)
    #    ls.append(np.sqrt(N) * (l - l_mle))
    #    hs.append(np.sqrt(N) * (h - h_mle))
    #    ss.append(np.sqrt(N) * (s - s_mle))
    #    #if ls[-1] >= 20:
    #    #    print(i)

    ##sns.distplot(ls)
    #sns.jointplot(ls, hs, kind="kde");
    #plt.tight_layout()
    #plt.show()
    #plt.close()

    #sns.distplot(ss);
    #plt.tight_layout()
    #plt.show()

    #sample = RSD.rvs(low=l, high=h, scale=s, size=N, random_state=24)
    #l_mle, h_mle, s_mle, _ = RSD.fit(sample)

    #print(l_mle, h_mle, s_mle)

    from sklearn.datasets import load_iris

    fig, axs = plt.subplots(2, 2)

    for i, ax in enumerate([ax for ax_row in axs for ax in ax_row]):
        sample = load_iris()["data"][:, i]
        l, h, s, _ = RSD.fit(sample)

        xs = np.linspace(l - 5*s - 1, h + 5*s + 1, 1000)

        ys = np.array(list(map(lambda x: RSD.pdf(x, l, h, s), xs)))
        ax.plot(xs, ys)

        ys = np.array(list(map(lambda x: RSD.pdf(x, l, h, s), sample)))
        ax.plot(sample, ys, "x")

        ax.set_title(load_iris()["feature_names"][i])

    plt.tight_layout()
    plt.savefig("iris.pdf")
    #print(l1, h1, s1)
    #print(RSD.NLL(sample, l1, h1, s1))

    #l,h,s = l1,h1,s1


    #ys = np.array(list(map(lambda x: np.sum(sample <= x) / len(sample), xs)))
    #plt.plot(xs, ys)

    #ys = np.array(list(map(lambda x: RSD.cdf(x, l, h, s), xs)))
    #plt.plot(xs, ys)


    #plt.show()
