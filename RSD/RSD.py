import numpy as np
from scipy import stats

from MLE_core.MLE_core import minimize_NLL


class RSD:
    @staticmethod
    def cdf(x, low=-1, high=1, scale=1):
        '''
        Cumulative distribution function at point x
        '''
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
        '''
        Probability density function at point x
        '''
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
        '''
        Generate random sample
        '''
        
        if np.isclose(scale, 0):
            return stats.uniform.rvs(low, high - low, size=size, random_state=random_state)

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
    def fit(data, weights=None, already_sorted=False):
        '''
        Perform maximum likelihood estimation.
        By default, all given points have equal weights.
        Returns optimal values of low, high, scale and corresponding NLL value.
        '''
        if weights is None:
            weights = np.ones(len(data))

        if not already_sorted:
            data = np.sort(data)

        return minimize_NLL(data, weights)

    @staticmethod
    def NLL(data, low=-1, high=1, scale=1, weights=None):
        N = len(data)
        if weights is None:
            weights = np.ones(N)
        r_sum = np.sum(weights)

        if np.isclose(scale, 0):
            return r_sum * np.log(high - low)
        else:
            result = r_sum * np.log(np.sqrt(2*np.pi) * scale + high - low)
            result += np.sum(weights * (np.maximum(0, data - high) - np.maximum(0, low - data))**2) / (2 * scale**2)
            return result
