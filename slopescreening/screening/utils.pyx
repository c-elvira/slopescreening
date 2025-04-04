# -*- coding: utf-8 -*-
# Based on https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# from libcpp cimport bool as bool_t

cimport cython

ctypedef np.npy_bool bool_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def GAP_Ptest_mixed_impl(
    # np.ndarray[np.npy_double, ndim=1] vec_f,
    np.ndarray[np.npy_double, ndim=1] vec_gammas,
    double lbd,
    np.ndarray[np.npy_double, ndim=1] Atcabs,
    double radius,
    np.ndarray[np.int64_t, ndim=1] index,
    ):

    cdef int n = Atcabs.shape[0]
    cdef int k, q, r
    cdef bool_t bool_exit = False
    cdef double tau = 0

    # arange because I want entries 0, 1 being 0 and 1!
    cdef np.ndarray[np.int64_t, ndim=1] vec_q_star = np.arange(n+1, dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim=1] vec_r_star = np.arange(n+1, dtype=np.int)

    # Sortie
    cdef np.ndarray[np.npy_bool, ndim=1] calI_screen = np.zeros(n, dtype=np.bool)

    # 2. Precomputing quantities
    cdef np.ndarray[np.npy_double, ndim=1] vec_f = np.cumsum(
        lbd * vec_gammas[::-1] - Atcabs[index[::-1]] - radius
    )[::-1]

    for k in range(2, n+1):
        #+1 is to match paper index
        vec_q_star[k] = k \
            if vec_f[k-1] - vec_f[k-2]  >  lbd * (vec_gammas[k-1] - vec_gammas[k-2]) \
            else vec_q_star[k-1]

        vec_r_star[k] = k \
            if vec_f[k-1] >  vec_f[k-2] \
            else vec_r_star[k-1]

    # 3. Tests
    for k in range(n, 0, -1):
        q = vec_q_star[k]
        while q >= 1:
            # Implicit definition of the best value of p given q
            r = vec_r_star[q]

            # Evaluation of the threshold
            tau = vec_f[r-1] - vec_f[q-1] + (lbd * vec_gammas[q-1] - radius)

            # Test
            if Atcabs[index[k-1]] >= tau:
                bool_exit = True
                break

            # Next critical point
            q = vec_q_star[r-1]

        if bool_exit:
            break

    calI_screen[index[k:]] = True
    return calI_screen


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def gap_kappa_test(
    np.ndarray[np.npy_double, ndim=1] vec_cumsum_gammas,
    np.ndarray[np.npy_double, ndim=1] vec_kappa_q,
    double lbd,
    np.ndarray[np.npy_double, ndim=1] Atcabs,
    double radius,
    np.ndarray[np.int64_t, ndim=1] index,
    ):

    cdef int n = Atcabs.shape[0]
    cdef int q, l_q
    cdef int l_min = 0
    cdef double bound

    # Sortie
    cdef np.ndarray[np.npy_bool, ndim=1] calI_screen = np.zeros(n, dtype=np.bool)

        # Cédric' notations compliant
    cdef np.ndarray[np.npy_double, ndim=1] sigmas = np.zeros(n)
    sigmas[1:] = np.cumsum(Atcabs[index[:-1]])

    for q in range(n-1, -1, -1):
        bound = lbd * vec_cumsum_gammas[q] - vec_kappa_q[q] * radius - sigmas[q]

        if Atcabs[index[q]] < bound:
            l_q = 0
        elif Atcabs[index[n-1]] >= bound:
            return calI_screen
        else:
            l_q = q + np.argmax(Atcabs[index[q:]] < bound)

        l_min = max(l_min, l_q)

    calI_screen[index[l_min:]] = True
    return calI_screen
