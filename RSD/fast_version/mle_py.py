import numpy as np

import quartic_solver


def traverse_extremal_chain(data, weights=None):
    '''
    Traverse the polygonal chain containing MLE extremum
    and calculate coefficients required for finding extremum.
    Note: input data vector should be already sorted (ascending)
    '''
    N = len(data)
    if weights is None:
        weights = np.ones(N)
    r_sum = np.sum(weights)
   
    i = 0
    u, v = 0, N - 1
    l_cur, h_cur = data[0], data[-1]
    r_l, r_h = weights[0], weights[-1]
    x_l, x_h = weights[0] * data[0], weights[-1] * data[-1]
    x_sq_l, x_sq_h = weights[0] * data[0]**2, weights[-1] * data[-1]**2
    
    a0 = np.zeros(N - 1, dtype=np.float64)
    b0 = np.zeros(N - 1, dtype=np.float64)
    c0 = np.zeros(N - 1, dtype=np.float64)
    d0 = np.zeros(N - 1, dtype=np.float64)
    e0 = np.zeros(N - 1, dtype=np.float64)

    while v - u >= 1:
        if v - u > 1:
            l_next = min(data[u + 1], (x_l + x_h - r_h * data[v - 1]) / r_l)
            h_next = max(data[v - 1], (x_l + x_h - r_l * data[u + 1]) / r_h)
        else:
            l_next = (x_l + x_h) / (r_l + r_h)
            h_next = l_next
        
        a = 1 / r_l + 1 / r_h
        b = x_h / r_h - x_l / r_l
        c = x_sq_l + x_sq_h - x_l**2 / r_l - x_h**2 / r_h
        
        a0[i] = r_sum * a**2
        b0[i] = -2 * np.pi * b
        c0[i] = 2 * c * (r_sum * a - np.pi)
        d0[i] = 0
        e0[i] = r_sum * c**2

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
        i += 1
    
    r1, r2, r3, r4 = quartic_solver.multi_quartic(a0, b0, c0, d0, e0)
    print(r1[-1], r2[0], r3[0], r4[0])
