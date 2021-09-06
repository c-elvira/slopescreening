# -*- coding: utf-8 -*-
import unittest
import numpy as np

import src.utils as utils
from src.dictionaries import generate_dic
from src.solver.slope import slope_gp

class TestUtilsModule(unittest.TestCase):

   def test_coherence(self):
      """ Compute the coherence manually and compare it 
      to the value output by the module
      """

      # 1. Create dictionary
      m = 20
      n = 50
      matA = generate_dic("gaussian", m, n, True)

      # 2. Compute coherence with module
      mu = utils.compute_coherence(matA)

      # 3. Compute coherence with brute force
      mu_prime = np.max(np.abs((matA.T @ matA) - np.eye(n)))

      # 4. Test
      self.assertAlmostEqual(mu, mu_prime, 14)


   def test_cound_coherence_function(self):
      """ Compute the coherence manually and compare it 
      to the value output by the module
      """

      # 1. Create dictionary
      m = 20
      n = 50
      matA = generate_dic("gaussian", m, n, True)

      # 2. Compute coherence with module
      mu = utils.compute_coherence(matA)
      coherence_func = utils.compute_coherence_function(matA)

      # 3. Test
      self.assertAlmostEqual(mu, coherence_func[1], 14)
      self.assertTrue( (coherence_func <= mu * np.arange(n)).all() )


   def test_coherences_pulse(self):
      """
         test on analytic dictionary defined in tropp's paper "greed is good"
         (unnormalzied dictionary)
      """

      # 1. Create dictionary
      m = 50
      n = 30
      matA = np.zeros((m, n))

      beta = .5
      for j in range(n):
         for i in range(j, m):
            matA[i, j] = np.sqrt(1 - beta**2) * beta**(i-j)

      # 2. Compute coherence with module
      mu = utils.compute_coherence(matA)
      mu_star = matA[:, 0] @ matA[:, 1]

      # 3. Test
      self.assertAlmostEqual(mu, mu_star, 14)


   def test_get_lambda_max(self):
      """
      """

      # 1. Create problem
      m = 20
      n = 50
      matA = generate_dic("gaussian", m, n, True)
      vecy = np.random.randn(m)
      vec_gammas = np.linspace(0, 1, n)[::-1]

      # 2. Compute lambda_max
      lbd_max = utils.get_lambda_max(vecy, matA, vec_gammas)

      # 3. Eval solution of slope problem
      out = slope_gp(vecy, matA, lbd_max, vec_gammas)

      # 4. Assert that the zero vector is solution
      self.assertTrue( (out["sol"] == np.zeros(n)).all() )


if __name__ == '__main__':
   unittest.main()