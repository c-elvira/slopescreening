# -*- coding: utf-8 -*-
# Based on https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
# https://cython.readthedocs.io/en/latest/src/tutorial/memory_allocation.html
import numpy as np

from slopescreening.screening.ascreening import AbstractGapScreening


# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# from libcpp cimport bool as bool_t

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

ctypedef np.npy_bool bool_t


cdef class GapTestAll:
   """ Generalized test
   """

   cdef double *vec_cumsum_gammas
   cdef char* name

   @cython.boundscheck(False) # turn off bounds-checking for entire function
   @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def __init__(
      self, np.ndarray[np.npy_double, ndim=1] vec_gammas
   ):
      
      cdef int n = vec_gammas.shape[0]
      cdef int i

      self.vec_cumsum_gammas = <double*> PyMem_Malloc((n+1) * sizeof(double))
      self.vec_cumsum_gammas[0] = 0

      for i in range(n):
         self.vec_cumsum_gammas[i+1] = self.vec_cumsum_gammas[i] + vec_gammas[i]

      # Adding zero at the beginning
      # cdef np.ndarray[np.npy_double, ndim=1] buf = np.zeros(vec_cumsum_gammas.shape[0]+1)
      # buf[1:] = vec_cumsum_gammas

      # self.vec_cumsum_gammas = &buf[0]

      self.name = "GAP test-all (cython)"


   def __dealloc__(self):
      PyMem_Free(self.vec_cumsum_gammas)  # no-op if self.data is NULL


   @cython.boundscheck(False) # turn off bounds-checking for entire function
   @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def get_name(self):
      return self.name

   @cython.boundscheck(False) # turn off bounds-checking for entire function
   @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def get_legend_name(self):
      return "all"

   @cython.boundscheck(False) # turn off bounds-checking for entire function
   # @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def apply_test(self, 
      np.ndarray[np.npy_double, ndim=1] Atcabs, 
      double gap, 
      double lbd, 
      np.ndarray[np.npy_double, ndim=1] vec_gammas,
      double coeff_dual_scaling =1.,
      double offset_radius=0., 
      np.ndarray[long, ndim=1] index=None
   ):
      """ Apply the Gap safe sphere screening test

      Implementation of the safe screening test
      
      Parameters
      ----------
      Atcabs : np.ndarray
         vector matA.T @ vecc where vec is dual admissible in absolute value
         size [n]
      gap : positive float
         duality gap
      lbd : positive float
         regularization parameters
      vec_gammas : np.ndarray
         slope parameters
         size [n,]
      coeff_dual_scaling : positif float
         If coeff_dual_scaling is not feasible, dual scaling factor
         such taht vecu / coeff_dual_scaling os dual feasible
         Here for code optimization purposes
         Default value is 1. (vecu is feasible)
      offset_radius : float
         additive term added to the redius
         default is 0
      index : np.ndarray
         Array of indices that sort Atu in absolute value
         default is None
      
      Returns
      -------
      calI_screen : np.ndarray
         vector of boolean
         True if screening test passes and False otherwise
         size [n,]
      """

      # !!! Important remark !!!
      # all indexes below range from 1 to n (instead of 0 to n-1)
      # in order to match the paper indexation rules

      cdef double radius  = coeff_dual_scaling * np.sqrt(2 * gap) + offset_radius
      # cdef double lbd_aug = coeff_dual_scaling * lbd
      cdef np.ndarray[np.npy_double, ndim=1] coeff_lbd_gamma = coeff_dual_scaling * lbd * vec_gammas

      cdef int n = Atcabs.shape[0]
      cdef int k, q, r
      cdef double tau = 0


      # arange because I want entries 0, 1 being 0 and 1!
      cdef np.ndarray[np.int_t, ndim=1] vec_p_star = np.arange(n+1, dtype=long)
      cdef np.ndarray[np.int_t, ndim=1] vec_q_star = np.arange(n+1, dtype=long)

      # output
      cdef np.ndarray[np.npy_bool, ndim=1] calI_screen = np.zeros(n, dtype=bool)

      # 1. Sort in descending order
      if index is None:
         index = np.argsort(Atcabs)[::-1]

      # 2. Precomputing quantities
      cdef np.ndarray[np.npy_double, ndim=1] vec_f = np.cumsum(
         coeff_lbd_gamma[::-1] - Atcabs[index[::-1]] - radius
      )[::-1]

      cdef double best_bound_q = vec_f[0] - coeff_lbd_gamma[0]
      cdef double curr_bound_q = 0.
      cdef double best_bound_p = vec_f[0]
      cdef double curr_bound_p = 0.
      for k in range(2, n+1):
         #+1 is to match paper index

         # 1. Evaluate p*
         curr_bound_p = vec_f[k-1]
         if curr_bound_p > best_bound_p:
            best_bound_p = curr_bound_p
            vec_p_star[k] = k
         else:
            vec_p_star[k] = vec_p_star[k-1]

         # 1. Evaluate q*
         curr_bound_q = vec_f[k-1] - coeff_lbd_gamma[k-1]
         if curr_bound_q > best_bound_q:
            best_bound_q = curr_bound_q
            vec_q_star[k] = k
         else:
            vec_q_star[k] = vec_q_star[k-1]

      # 3. Tests
      for l in range(n, 0, -1):
         q = vec_q_star[l]
         while q >= 1:
            # Implicit definition of the best value of p given q
            p = vec_p_star[q]

            # Evaluation of the threshold
            tau = vec_f[p-1] - vec_f[q-1] + (coeff_lbd_gamma[q-1] - radius)

            # Test
            if Atcabs[index[l-1]] >= tau:
               calI_screen[index[l:]] = True
               return calI_screen

            # Next critical point
            q = vec_q_star[p-1]


      calI_screen[index[l:]] = True
      return calI_screen