# -*- coding: utf-8 -*-
# Based on https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
# https://cython.readthedocs.io/en/latest/src/tutorial/memory_allocation.html
import numpy as np

from src.screening.ascreening import AbstractGapScreening


# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# from libcpp cimport bool as bool_t

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

ctypedef np.npy_bool bool_t


#(AbstractGapScreening)

cdef class GAP_RQtest:
   """ Generalized test
   """

   cdef double *vec_cumsum_gammas
   cdef long *vec_rq

   @cython.boundscheck(False) # turn off bounds-checking for entire function
   @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def __init__(
      self, 
      np.ndarray[np.npy_double, ndim=1] vec_gammas,
      np.ndarray[long, ndim=1] vec_rq,
   ):
      
      cdef int n = vec_gammas.shape[0]
      cdef int i

      self.vec_cumsum_gammas = <double*> PyMem_Malloc((n+1) * sizeof(double))
      self.vec_cumsum_gammas[0] = 0

      self.vec_rq = <long*> PyMem_Malloc(n * sizeof(long))

      for i in range(n):
         self.vec_cumsum_gammas[i+1] = self.vec_cumsum_gammas[i] + vec_gammas[i]
         self.vec_rq[i] = vec_rq[i]


   def __dealloc__(self):
      PyMem_Free(self.vec_cumsum_gammas)  # no-op if self.data is NULL
      PyMem_Free(self.vec_rq)

   def get_name(self):
      return "p-test cython"


   @cython.boundscheck(False) # turn off bounds-checking for entire function
   @cython.wraparound(False)  # turn off negative index wrapping for entire function
   def apply_test(self, 
      np.ndarray[np.npy_double, ndim=1] Atcabs, 
      double gap, 
      double lbd, 
      np.ndarray[np.npy_double, ndim=1] vec_gammas, 
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

      cdef double radius = np.sqrt(2 * gap) + offset_radius

      cdef int n = Atcabs.shape[0]
      cdef int k, l, q, r_q
      cdef double lhs

      # right-hand side
      cdef np.ndarray[np.npy_double, ndim=1] vec_kappaqr
      vec_kappaqr = np.zeros(n+1, dtype=np.double)

      for q in range(1, n+1):
         r_q = self.vec_rq[q-1]

         vec_kappaqr[q] += lbd * (self.vec_cumsum_gammas[q] - self.vec_cumsum_gammas[r_q-1])
         vec_kappaqr[q] -= (q - r_q + 1) * radius

      # output
      cdef np.ndarray[np.npy_bool, ndim=1] calI_screen = np.ones(n, dtype=bool)

      # 1. Sort in descending order
      if index is None:
         index = np.argsort(Atcabs)[::-1]

      cdef np.ndarray[np.npy_double, ndim=1] vec_cumsumAtcabs = np.zeros(n+1, dtype=np.double)
      vec_cumsumAtcabs[1:] = np.cumsum(Atcabs[index])

      for l in range(n):
         index_l = index[l]

         for q in range(1, n+1): # Index are shifted by 1
            r_q = self.vec_rq[q-1]
            # lhs = 0

            if r_q >= q:
               lhs = Atcabs[index_l]

            # elif index_l < index[r_q-1] or index_l >= index[q-1]:
            elif l < r_q or l > q-1:
               #In this calse |A_{\l}c|_[k] =|Atc|_[k] for the range of k 
               lhs = Atcabs[index_l] + vec_cumsumAtcabs[q-1] - vec_cumsumAtcabs[r_q-2]

               # for k in range(r_q-1, q-1):
               #    lhs += Atcabs[index[k]]

            else:
               # print("ici else")
               lhs = vec_cumsumAtcabs[q] - vec_cumsumAtcabs[r_q-2]
               # for k in range(r_q-1, q):
               #    lhs += Atcabs[index[k]]

            if lhs >= vec_kappaqr[q]:

               calI_screen[index_l] = False
               break
   
      return calI_screen