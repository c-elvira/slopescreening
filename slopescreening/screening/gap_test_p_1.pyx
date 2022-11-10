# -*- coding: utf-8 -*-
# Based on https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
import numpy as np

from slopescreening.screening.ascreening import AbstractGapScreening


# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# from libcpp cimport bool as bool_t

cimport cython

ctypedef np.npy_bool bool_t

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


#(AbstractGapScreening)

cdef class GapTestPequalOne:
   """ Generalized test
   """

   # cdef double[:] vec_cumsum_gammas
   # cdef double[:] vec_kappa_q

   cdef double* vec_cumsum_gammas
   cdef double* vec_kappa_q
   cdef char* name

   # cdef np.ndarray vec_cumsum_gammas
   # cdef np.ndarray vec_kappa_q

   @cython.boundscheck(False) # turn off bounds-checking for entire function
   @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def __init__(
      self, 
      np.ndarray[np.npy_double, ndim=1] vec_gammas,
      np.ndarray[np.npy_double, ndim=1] vec_coherence_function,
      ):

      cdef int n = vec_gammas.shape[0]
      # cdef np.ndarray[np.npy_double, ndim=1] buf1
      # cdef np.ndarray[np.npy_double, ndim=1] buf2

      # buf1 = np.cumsum(vec_gammas).copy()
      # self.vec_cumsum_gammas = &buf1[0]

      # buf2 = np.sqrt( 
      #    (1. + vec_coherence_function) * np.arange(1, n+1, dtype=np.double)
      # ).copy()
      # self.vec_kappa_q = &buf2[0]

      self.vec_cumsum_gammas = <double*> PyMem_Malloc(n * sizeof(double))
      self.vec_cumsum_gammas[0] = vec_gammas[0]

      self.vec_kappa_q = <double*> PyMem_Malloc(n * sizeof(double))

      cdef int i = 0
      for i in range(1, n):
         self.vec_cumsum_gammas[i] = self.vec_cumsum_gammas[i-1] + vec_gammas[i]

      for i in range(n):
         self.vec_kappa_q[i] = np.sqrt( 
         (1. + vec_coherence_function[i]) * (i + 1.)
      )

      # # self.vec_cumsum_gammas = np.cumsum(vec_gammas)
      # self.vec_kappa_q = np.sqrt( 
      #    (1. + vec_coherence_function) * np.arange(1, n+1, dtype=np.double)
      # ).copy()

      self.name = "Gap test p=1 (cython)"


   def __dealloc__(self):
      PyMem_Free(self.vec_cumsum_gammas)  # no-op if self.data is NULL
      PyMem_Free(self.vec_kappa_q)  # no-op if self.data is NULL


   @cython.boundscheck(False) # turn off bounds-checking for entire function
   @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def get_name(self):
      return self.name

   @cython.boundscheck(False) # turn off bounds-checking for entire function
   @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def get_legend_name(self):
      return "$p_q=1\\;\\forall q$"

   @cython.boundscheck(False) # turn off bounds-checking for entire function
   @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def apply_test(self, 
      np.ndarray[np.npy_double, ndim=1] Atcabs, 
      double gap, 
      double lbd, 
      np.ndarray[np.npy_double, ndim=1] vec_gammas,
      double coeff_dual_scaling = 1., 
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

      cdef int n = Atcabs.shape[0]
      cdef int q, l_q
      cdef int l_min = 0

      cdef double bound
      cdef double radius = coeff_dual_scaling * np.sqrt(2 * gap) + offset_radius
      cdef double coeff_lbd = coeff_dual_scaling * lbd

      cdef np.ndarray[np.npy_double, ndim=1] sigmas = np.zeros(n+1)

      # output
      cdef np.ndarray[np.npy_bool, ndim=1] calI_screen = np.zeros(n, dtype=bool)

      # 1. Sort in descending order
      if index is None:
         index = np.argsort(Atcabs)[::-1]

         # Cédric' notations compliant
      sigmas[1:] = np.cumsum(Atcabs[index])

      for q in range(n-1, -1, -1):
         bound = coeff_lbd * self.vec_cumsum_gammas[q] - self.vec_kappa_q[q] * radius - sigmas[q]

         if Atcabs[index[q]] < bound:
            l_q = 0

         elif Atcabs[index[n-1]] >= bound:
            return calI_screen

         else:
            l_q = q + np.argmax(Atcabs[index[q:]] < bound)

         l_min = max(l_min, l_q)

      calI_screen[index[l_min:]] = True
      return calI_screen