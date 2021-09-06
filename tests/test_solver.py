# -*- coding: utf-8 -*-
import unittest
import numpy as np

import src.utils as utils
from src.dictionaries import generate_dic
from src.solver.slope import slope_gp
from src.solver.parameters import SlopeParameters


class TestSolver(unittest.TestCase):

   def test_gp_cost_decrease(self):
      """ Run a non accelerated proximal gradient algorithm
      assess that the cost function decreaes
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
      algParameters = SlopeParameters()
      algParameters.max_it = 1000
      algParameters.accelerated = False
      out = slope_gp(vecy, matA, .5*lbd_max, vec_gammas, algParameters)

      # 4. Assert that the zero vector is solution
      vec_cost = out["cost_function"]
      vec_diff = (vec_cost[1:] - vec_cost[:-1])

         # 1e-15 due to machine precision error
      self.assertTrue( (vec_diff <= 1e-14).all() )

if __name__ == '__main__':
   unittest.main()