# -*- coding: utf-8 -*-
import unittest
import numpy as np

from src.dictionaries import generate_dic
from src.solver.slope import primal_func, dual_func, slope_gp
from src.solver.parameters import SlopeParameters
from src.screening.singletest import GapSphereSingleTest
from src.screening.gap_ptest import GAP_Ptest
import src.utils as utils

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
      vec_gammas = np.linspace(.5, 1, n)[::-1]

      # 2. Compute lambda_max
      lbd_max = utils.get_lambda_max(vecy, matA, vec_gammas)
      lbd = .6 * lbd_max

      # 3. Eval solution of slope problem
      algParameters = SlopeParameters()
      algParameters.gap_stopping = 1e-12
      algParameters.max_it = 100000
      algParameters.accelerated = False
      out = slope_gp(vecy, matA, .5*lbd_max, vec_gammas, algParameters)

      vecu = vecy - matA @ out["sol"]
      beta_dual = np.sort(np.abs(matA.T @ vecu))[::-1]
      beta_dual = np.cumsum(beta_dual) / np.cumsum(lbd * vec_gammas)
      vecu /= np.max(beta_dual)
      Atu = matA.T @ vecu

      pval = primal_func(vecy, matA @ out["sol"], out["sol"], lbd, vec_gammas)
      dval = dual_func(vecy, np.linalg.norm(vecy, 2)**2, vecu)
      gap = np.abs(pval - dval)

      # 4. Start screening test
      test1 = GapSphereSingleTest()
      test2 = GAP_Ptest(np.cumsum(vec_gammas))

      out1 = test1.apply_test(Atu, gap, lbd, vec_gammas)
      out2 = test2.apply_test(Atu, gap, lbd, vec_gammas)

         # 1e-15 due to machine precision error
      self.assertTrue( (out2 >= out1).all() )

if __name__ == '__main__':
   unittest.main()