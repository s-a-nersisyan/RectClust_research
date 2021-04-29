cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport abs, sqrt, log, isnan, M_PI, INFINITY

# Import math functions to work with complex numbers
cdef extern from "complex.h":
    np.complex128_t csqrt(np.complex128_t)
    np.complex128_t ccos(np.complex128_t)
    np.complex128_t cacos(np.complex128_t)
    np.complex128_t cpow(np.complex128_t, np.complex128_t)

# Tolerance for comparing numbers with zeros
cdef np.float64_t tol = 1e-8


@cython.cdivision(True)
cdef (np.complex128_t, np.complex128_t) solve_quadratic_equation(
    np.complex128_t a0,
    np.complex128_t b0,
    np.complex128_t c0
):
    '''
    Solve the quadratic equation
    a0*x^2 + b0*x + c0
    '''
    # a0 -> 1
    cdef np.complex128_t a = b0 / a0
    cdef np.complex128_t b = c0 / a0

    a0 = -0.5*a
    cdef np.complex128_t delta = a0*a0 - b
    cdef np.complex128_t sqrt_delta = csqrt(delta)

    return (a0 - sqrt_delta, a0 + sqrt_delta)


@cython.cdivision(True)
cdef np.complex128_t signed_cubic_root(np.complex128_t x):
    '''
    Find cubic root of x with correct sign
    '''
    if x.real >= 0:
        return cpow(x, 1.0 / 3.0)
    else:
        return -cpow(-x, 1.0 / 3.0)


@cython.cdivision(True)
cdef np.complex128_t cubic_equation_real_root(
    np.complex128_t a0,
    np.complex128_t b0,
    np.complex128_t c0,
    np.complex128_t d0
):
    '''
    Find real root of the cubic equation
    a0*x^3 + b0*x^2 + c0*x + d0
    '''
    # a0 -> 1
    cdef np.complex128_t a = b0 / a0
    cdef np.complex128_t b = c0 / a0
    cdef np.complex128_t c = d0 / a0

    cdef np.complex128_t a13 = a / 3.0
    cdef np.complex128_t a2 = a13*a13
    cdef np.complex128_t f = b / 3.0 - a2
    cdef np.complex128_t g = a13 * (2*a2 - b) + c
    cdef np.complex128_t h = 0.25*g*g + f*f*f
    
    cdef np.complex128_t j, k, m, sqrt_h
    if f == g and g == h and h == 0:
        return -signed_cubic_root(c)
    elif h.real <= 0:
        j = csqrt(-f)
        k = cacos(-0.5*g / (j*j*j))
        m = ccos(k / 3.0)
        return 2*j*m - a13
    else:
        sqrt_h = csqrt(h)
        return signed_cubic_root(-0.5*g + sqrt_h) + signed_cubic_root(-0.5*g - sqrt_h) - a13


@cython.cdivision(True)
cdef (np.complex128_t, np.complex128_t, np.complex128_t, np.complex128_t) solve_quartic_equation(
    np.complex128_t a0, 
    np.complex128_t b0, 
    np.complex128_t c0, 
    np.complex128_t d0,
    np.complex128_t e0
):
    '''
    Solve the quartic equation 
    a0*x^4 + b0*x^3 + c0*x^2 + d0*x + e0
    '''
    # a0 -> 1
    cdef np.complex128_t a = b0 / a0
    cdef np.complex128_t b = c0 / a0
    cdef np.complex128_t c = d0 / a0
    cdef np.complex128_t d = e0 / a0

    # Some repeating variables
    a0 = 0.25*a
    cdef np.complex128_t a02 = a0*a0

    # Coefficients of subsidiary cubic euqtion
    cdef np.complex128_t p = 3*a02 - 0.5*b
    cdef np.complex128_t q = a*a02 - b*a0 + 0.5*c
    cdef np.complex128_t r = 3*a02*a02 - b*a02 + c*a0 - d

    # One root of the cubic equation
    cdef np.complex128_t z0 = cubic_equation_real_root(1, p, r, p*r - 0.5*q*q)

    # Additional variables
    cdef np.complex128_t s = csqrt(2*p + 2*z0.real)
    cdef np.complex128_t t
    if s == 0:
        t = z0*z0 + r
    else:
        t = -q / s

    # Compute roots of quadratic equations
    cdef np.complex128_t r0, r1, r2, r3
    
    r0, r1 = solve_quadratic_equation(1, s, z0 + t)
    r2, r3 = solve_quadratic_equation(1, -s, z0 - t)
    
    return (r0 - a0, r1 - a0, r2 - a0, r3 - a0)


@cython.cdivision(True)
cdef (np.float64_t, np.float64_t, np.float64_t, np.float64_t) optimize_NLL_on_segment(
    np.float64_t r_sum, np.float64_t r_l, np.float64_t r_h,
    np.float64_t x_l, np.float64_t x_h,
    np.float64_t x_sq_l, np.float64_t x_sq_h,
    np.float64_t l_cur, np.float64_t l_next,
    long int iternum,
    np.float64_t n0, np.float64_t s0
):
    '''
    Minimize negative log-likelihood on a given segment
    '''
    cdef np.float64_t t_cur = r_l * l_cur - x_l
    cdef np.float64_t t_next = r_l * l_next - x_l
    
    # Temporary variables
    cdef np.float64_t t, l, h, s, NLL
    cdef np.float64_t at, a_inv
    cdef np.float64_t b1, b0
    cdef np.float64_t a3, a2, a0
    cdef np.complex128_t r0, r1, r2, r3
    
    # Local minimum variables
    cdef np.float64_t t_loc_min, s_loc_min, NLL_loc_min
    NLL_loc_min = INFINITY

    # Auxiliarly constants
    cdef np.float64_t min_r_l_h = min(r_l, r_h)
    cdef np.float64_t max_r_l_h = max(r_l, r_h)
    cdef np.float64_t b = x_h / r_h - x_l / r_l
    cdef np.float64_t c = x_sq_l + x_sq_h - x_l**2 / r_l - x_h**2 / r_h
    
    # First, solve the quartic equation for critical values of t
    a_inv = min_r_l_h / (1 + min_r_l_h/max_r_l_h)
    a3 = -2 * M_PI * b * a_inv**2 / (r_sum + n0 + 2)
    a2 = 2 * (c + n0*s0**2) * a_inv * (1 - M_PI * a_inv / (r_sum + n0 + 2))
    a0 = ((c + n0*s0**2) * a_inv)**2
    r0, r1, r2, r3 = solve_quartic_equation(1, a3, a2, 0, a0)
    
    # Critical points = boundary + roots of the derivative
    cdef np.complex128_t critical_t[6]
    critical_t[:] = [t_cur, t_next, r0, r1, r2, r3]
    
    cdef int i = 0
    for i in range(6):
        # We do not want complex roots
        if abs(critical_t[i].imag) > tol:
            continue
        t = critical_t[i].real

        # We do not want t = nan
        if isnan(t):
            continue
        
        # Roots should lie within [t_cur, t_next] interval
        if i > 1 and (t < t_cur or t > t_next):
            continue
        
        l = (t + x_l) / r_l
        h = (x_l + x_h - r_l*l) / r_h
        at = (1 + min_r_l_h/max_r_l_h) * (r_l / min_r_l_h * l - x_l / min_r_l_h)
        
        if i == 0 or i == 1 or abs(t) <= tol:
            # Solve the cubic equation for s
            b1 = -(at*t + c + n0*s0**2) / (r_sum + n0 + 2)
            b0 = (h - l) / sqrt(2*M_PI) * b1
            s = cubic_equation_real_root(1, 0, b1, b0).real
        else:
            s = (at*t + c + n0*s0**2) / sqrt(2*M_PI) / t
        
        if iternum == 0 and i == 0 and (n0 == 0 or s0 == 0):
            s = 0
            NLL = (r_sum + n0 + 2) * log(b)
        elif s < 0 or isnan(s):
            NLL = INFINITY
        else:
            NLL = (r_sum + n0 + 2) * log(sqrt(2*M_PI) * s - at + b) + (at*t + c + n0*s0**2) / (2 * s**2)
        
        if NLL < NLL_loc_min:
            t_loc_min = t
            s_loc_min = s
            NLL_loc_min = NLL

    cdef np.float64_t l_loc_min = (t_loc_min + x_l) / r_l
    cdef np.float64_t h_loc_min = (x_l + x_h - r_l * l_loc_min) / r_h

    return (l_loc_min, h_loc_min, s_loc_min, NLL_loc_min)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef (np.float64_t, np.float64_t, np.float64_t, np.float64_t) minimize_NLL(
        np.ndarray[np.float64_t, ndim=1] data, 
        np.ndarray[np.float64_t, ndim=1] weights,
        np.float64_t n0, np.float64_t s0
):
    '''
    Fast minimization of RSD negative log-likelihood.
    Briefly:
        - Traverse the polygonal chain containing the global minimum;
        - Find local minimum on each segment of the chain;
        - Take the global minimum.
    Note: input data vector should be already sorted (ascending).
    '''
    # Total number of points
    cdef long int N = data.shape[0]
    # Total weight
    cdef np.float64_t r_sum = np.sum(weights)
    
    # Global minimums (the results)
    cdef np.float64_t l_glob_min, h_glob_min, s_glob_min, NLL_glob_min
    NLL_glob_min = INFINITY
    
    # Ignore first and last entries with near-zero weights
    # Region indexers (u is a number of l axis, v is a number of h axis)
    cdef long int u = 0
    cdef long int v = N - 1

    while u < v:
        if abs(weights[u]) > tol and abs(weights[v]) > tol:
            break

        if abs(weights[u]) <= tol:
            u += 1
        
        if abs(weights[v]) <= tol:
            v -= 1
    
    N = v - u + 1

    # Solve the easy case: N = 0 or N = 1
    if N == 0 or N == 1:
        if N == 0:
            l_glob_min, h_glob_min = 0, 0
        else:
            l_glob_min, h_glob_min = data[u], data[u]

        s_glob_min = sqrt(n0 / (r_sum + n0 + 2)) * s0
        NLL_glob_min = (r_sum + n0 + 2) * log(sqrt(2*M_PI)*s_glob_min)
        NLL_glob_min += n0 * (s0/s_glob_min)**2 / 2
        
        return (l_glob_min, h_glob_min, s_glob_min, NLL_glob_min)
    
    # Global counter
    cdef long int i = 0
    
    # Coordinates of current segment's left-top point
    cdef np.float64_t l_cur = data[u]
    cdef np.float64_t h_cur = data[v]
    # Coordinates of current segment's bottom-right point (or left-top coordinates for the next segment)
    cdef np.float64_t l_next
    cdef np.float64_t h_next
    
    # Sum of weights for l and h
    cdef np.float64_t r_l = weights[u]
    cdef np.float64_t r_h = weights[v]
    
    # Weighted sum of x's for l and h
    cdef np.float64_t x_l = weights[u] * data[u]
    cdef np.float64_t x_h = weights[v] * data[v]
    
    # Weighted sum of x squares for l and h
    cdef np.float64_t x_sq_l = weights[u] * data[u]**2
    cdef np.float64_t x_sq_h = weights[v] * data[v]**2
    
    # Local minimums
    cdef np.float64_t l_loc_min, h_loc_min, s_loc_min, NLL_loc_min
    
    # Traverse extremal chain from l = min(data), h = max(data) to l = h
    while v - u >= 1:
        # First, determine coordinates l_next and h_next
        if v - u > 1:
            l_next = min(data[u + 1], (x_l + x_h - r_h * data[v - 1]) / r_l)
            h_next = max(data[v - 1], (x_l + x_h - r_l * data[u + 1]) / r_h)
        else:
            l_next = (x_l + x_h) / (r_l + r_h)
            h_next = l_next
        
        # Then, find local minimum on current segment
        l_loc_min, h_loc_min, s_loc_min, NLL_loc_min = optimize_NLL_on_segment(
            r_sum, r_l, r_h,
            x_l, x_h,
            x_sq_l, x_sq_h,
            l_cur, l_next,
            i,
            n0, s0
        )
        
        # Compare with current global minimum
        if NLL_loc_min <= NLL_glob_min:
            l_glob_min = l_loc_min
            h_glob_min = h_loc_min
            s_glob_min = s_loc_min
            NLL_glob_min = NLL_loc_min
        
        # Finally, move to the next segment
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
    
    return (l_glob_min, h_glob_min, s_glob_min, NLL_glob_min)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def minimize_NLL_matrix(
        np.ndarray[np.float64_t, ndim=2] X_sorted,
        np.ndarray[np.int64_t, ndim=2] X_argsort,
        np.ndarray[np.float64_t, ndim=2] r,
        np.float64_t n0, np.float64_t s0
):
    # Array sizes
    cdef long int n_samples = X_sorted.shape[0]
    cdef int n_features = X_sorted.shape[1]
    cdef int n_clusters = r.shape[1]
    
    # Results
    cdef np.ndarray[np.float64_t, ndim=2] low = np.empty((n_clusters, n_features))
    cdef np.ndarray[np.float64_t, ndim=2] high = np.empty((n_clusters, n_features))
    cdef np.ndarray[np.float64_t, ndim=2] scale = np.empty((n_clusters, n_features))
    cdef np.float64_t prior = 0

    # Temporary variables
    cdef np.ndarray[np.float64_t, ndim=1] data = np.empty(n_samples)
    cdef np.ndarray[np.float64_t, ndim=1] weights = np.empty(n_samples)
    cdef np.float64_t low_MLE, high_MLE, scale_MLE, NLL_min

    for j in range(n_features):
        for k in range(n_clusters):
            # Sort weights according to the j-th component of X
            for i in range(n_samples):
                data[i] = X_sorted[i, j]
                weights[i] = r[X_argsort[i, j], k]

            low_MLE, high_MLE, scale_MLE, NLL_min = minimize_NLL(data, weights, n0, s0)
            low[k, j] = low_MLE
            high[k, j] = high_MLE
            scale[k, j] = scale_MLE
            
            if scale_MLE != 0 and sqrt(2*M_PI)*scale_MLE + high_MLE - low_MLE > 0:
                prior += (n0 + 2) * log(sqrt(2*M_PI)*scale_MLE + high_MLE - low_MLE) + n0 * (s0 / scale_MLE)**2 / 2
            else:
                prior = INFINITY

    return (low, high, scale, prior)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def log_p_matrix(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.float64_t, ndim=2] low,
    np.ndarray[np.float64_t, ndim=2] high,
    np.ndarray[np.float64_t, ndim=2] scale
):
    '''
    Compute and return probability matrix ( log p_k(x_i) )
    '''
    cdef long int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_clusters = low.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty((n_samples, n_clusters))
    cdef long int i
    cdef int k
    cdef np.float64_t log_p, C

    for i in range(n_samples):
        for k in range(n_clusters):
            log_p = 0

            for j in range(n_features):
                if abs(scale[k, j]) < tol:
                    if low[k, j] <= X[i, j] and X[i, j] <= high[k, j]:
                        log_p += -log(high[k, j] - low[k, j])
                    else:
                        log_p = -INFINITY
                        break
                else:
                    log_p += -log(sqrt(2 * M_PI) * scale[k, j] + high[k, j] - low[k, j])
                    if X[i, j] < low[k, j]:
                        log_p += -(X[i, j] - low[k, j]) * (X[i, j] - low[k, j]) / scale[k, j] / scale[k, j] / 2
                    elif X[i, j] > high[k, j]:
                        log_p += -(X[i, j] - high[k, j]) * (X[i, j] - high[k, j]) / scale[k, j] / scale[k, j] / 2
            
            result[i, k] = log_p

    return result
