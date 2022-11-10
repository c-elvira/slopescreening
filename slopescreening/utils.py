# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import ndtri


def get_lambda_max(vecy, matA, vecgammas) -> float:
   """ Compute the highest value of parameter
   lambda such that the solution to the Slope
   problem is not the zero vector.

   (To be updated) Expression given in Eq.~(3.2) of our paper

   Parameters
   ----------
   vecy : np.ndarray
      observation vector
      size [m,]
   matA : np.ndarray
      dictionary matrix
      size [m, n]
   vecgammas : np.ndarray
      vector of Slope coefficients
      size [n]
   
   Returns
   -------
   lambdamax : float
      highest value of lambda such that 
      the solution of the Slope problem
      is nonzero
   """

   buf = np.sort(np.abs(matA.T @ vecy))[::-1]
   buf = np.cumsum(buf) / np.cumsum(vecgammas)

   return np.max(buf)


def gamma_sequence_generator(m, n, seq_type, alpha, log=False) -> np.ndarray:
   """Compute gamma sequence given parameters.
   
   The parametrization originates from [2,3] and are all 
   sumarized in [1, section 3.1].
   Three type are accepted.

   1) The oscar sequence defined by
   \\[
      \\gamma_j^{oscar} = 1 + alpha * (n - j)
   \\]

   2) The Benjamini–Hochberg (BH) sequence defined by
   \\[
      \\gamma_l^{BH} = \\phi^{-1} (1 - \\alpha\\frac{l}{n})
   \\]


   1) The Gaussian (G) sequence defined by
   \\[
      \\gamma_j^G = \\gamma_j^{BH} * \\sqrt{(1 + 1 / (m - j) * \\sum_{l < j}\\gamma_l^{G}^2)}
   \\]



   [1] Johan Larsson1, Małgorzata Bogdan and Jonas Wallin1
      ``The Strong Screening Rule for SLOPE'' 
      preprint arxiv (2020)
   
   [2] Howard D. Bondell and Brian J. Reich, 
       ``Simultaneous Regression Shrinkage, Variable Selection, 
       and Supervised Clustering of Predictors with OSCAR''
   
   [3] Małgorzata Bogdan et al. 
      ``LOPE - Adaptive Variable Selection via ConvexOptimization''.
      In:The annals of applied statistics9.3 (2015),
      pp. 1103–1140.issn: 1932-6157.doi: 10.1214/15-AOAS842.
      pmid: 26709357.
   
   Parameters
   ----------
   n : int
      size of the sequence
   seq_type : str
      type of sequence.
      Accepted type ['oscar', 'benjamini–hochberg', 'gaussian']
   alpha : float
      parameter of the sequence
   log : bool
      provide informations about sequences if true

   Returns
   -------
   vec_gamma : np.ndarray
      sequence of gamma

   Raises
   ------
   Exception
      if seq_type is invalid
   """
   assert(isinstance(n, int) and n>= 1)
   assert(isinstance(seq_type, str))


   if seq_type.lower() == 'lasso':
      vec_gammas = np.ones(n, dtype=np.float64)

   elif seq_type.lower() == 'antisparse':
      vec_gammas = np.zeros(n, dtype=np.float64)
      vec_gammas[0] = 1.

   elif seq_type.lower() == 'oscar':
      vec_gammas = 1. + alpha * (n - np.arange(1, n+1, dtype=np.float64))

   elif seq_type.lower() == 'oscar-lim':
      # beta_1 = alpha[0]
      # beta_2 = (alpha[1] - alpha[0]) / (n - 1.)
      beta_1 = (alpha[0] - n * alpha[1]) / (1. - n)
      beta_2 =   (alpha[1] - alpha[0])   / (1. - n)

      vec_gammas = beta_1 + beta_2 * np.arange(n, 0, -1, dtype=np.double) 
      vec_gammas[vec_gammas < 0] = 0.

      if log:
         print("OSCAR sequence with")
         print(f"\tbeta_1 = {beta_1}")
         print(f"\tbeta_2 = {beta_2}")

   elif seq_type.lower() == 'exp':
      vec_gammas =  1. - alpha[0] * np.exp(-alpha[1] * np.arange(n-1, -1, -1, dtype=np.float64))

   elif seq_type.lower() == 'exp-lim':
      # alpha[0] = \\gamma_1
      # alpha[1] = \\gamma_n

      assert(alpha[0] < 1)

      beta_2 = np.log( (1 - alpha[1]) / (1 - alpha[0]) ) / (n - 1.)
      beta_1 = (1 - alpha[0]) * np.exp(beta_2 * n)
      vec_gammas =  1. - beta_1 * np.exp(-beta_2 * np.arange(n-1, -1, -1, dtype=np.float64))

      if log:
         print("EXP sequence with")
         print(f"\tbeta_1 = {beta_1}")
         print(f"\tbeta_2 = {beta_2}")

   elif seq_type.lower() == 'benjamini–hochberg':
      vec_gammas =  ndtri(1. - alpha[0] * .5 * np.arange(1, n+1, dtype=np.float64) / n)

   elif seq_type.lower() == 'gaussian':
      vec_gammas = np.zeros(n)
      vec_gammas_BH =  ndtri(1. - alpha * (1 + np.arange(min(m, n), dtype=np.float64)) / (2 * min(m, n)))

      vec_gammas[0] = vec_gammas_BH[0]
      for j in range(1, min(m, n)):
         buf = vec_gammas_BH[j] * np.sqrt(
            1. + (np.sum(vec_gammas[:j]**2) / float(m - j - 1))
         )

         if buf > vec_gammas[j-1]:
            vec_gammas[j] = vec_gammas[j-1]

   else:
      raise Exception(f"seq_type not recognized -- {seq_type}")

   vec_gammas[vec_gammas < 0] = 0
   vec_gammas[np.isnan(vec_gammas)] = 0

   return vec_gammas


def compute_coherence(matA) -> float:
   """ Compute the coherence of a dictionary
    
   Parameters
   ----------
   matA : np.ndarray
      Dictionary
      size [m, n]
   
   Returns
   -------
   mu : float
      Highest correlation between two column
   """

   mu = 0.
   for i in range(1, matA.shape[1]):
      # ~~ nb multiplications ~~
      # A[:, i].T @ matA[:, :i]: m * i
      mu = max(
         np.max(np.abs(matA[:, i].T @ matA[:, :i])),
         mu
      )

   # nbmult += m * i
   return mu


def compute_coherence_function(matA) -> np.ndarray:
   """ A memory efficient implementation of the evaluation of the coherence function

   Parameters
   ----------
   matA : numpy.ndarray
      [m, n] dictionary

   Returns
   -------
   vec_coherence : numpy.ndarray
      coherence function of the dictionary
      entry 0 is equal to 0 (by convention)
      entry 1 is the coherence
      size [n,]
   """

      # size of dictionary
   n = matA.shape[1]

      # Coherence function, evaluated from 0 to n-1
   vec_coherence = np.zeros(n)

      # evaluation
   for j in range(n):
         # a. sorting correlation in reverse order
      corrs = np.sort(np.abs(matA.T @ matA[:, j]))[::-1]

         # b. remove the autocorrelation
      corrs = corrs[1:]

         # c. cumsum
      corrs = np.cumsum(corrs)

         # d. entry-wise maximum
      vec_coherence[1:] = np.maximum(corrs, vec_coherence[1:])

   return vec_coherence



if __name__ == '__main__':
   import dictionaries

   m = 2
   n = 4

   matA = dictionaries.generate_dic("gaussian", m, n, True)
   cohefun = compute_coherence_function(matA)

   print(cohefun)

   # Brute force computation
   matG = np.abs((matA.T @ matA) - np.eye(n))
   print(matG)
   for j in range(n):
      matG[:, j] = np.sort(np.abs(matG[:, j]))[::-1]

   print(matG)