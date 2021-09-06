import numpy as np

from src.solver.parameters import SlopeParameters, EnumLipchitzOptions

from src.screening.singletest import GapSphereSingleTest
from src.screening.gap_ptest import GAP_Ptest
from src.screening.kappa_test import Kappa_test

nb_algs = 3

def get_nb_algs(setup):

   return nb_algs


def get_alg_params(setup, vec_gammas, exact):

   list_params = []
   list_names  = []

   # --- 1. Xp ---
      # 1a. No screening
   params1 = SlopeParameters()

   list_params.append(params1)
   list_names.append("no screening")


   params2 = SlopeParameters()

   if exact:
      params2.screening2 = GapSphereSingleTest()
      params2.screening_it_div = 2.

   else:
      params2.screening1 = GapSphereSingleTest()

   list_params.append(params2)
   list_names.append("single test")

      # 1c. Single test screening
   params3 = SlopeParameters()

   params3.screening2 = GAP_Ptest(vec_gammas)
   params3.screening_it_div = 2.

   if not exact:
      params3.screening1 = GapSphereSingleTest()

   list_params.append(params3)
   list_names.append("p-test")


   # --- 2. Common parameters ---
   for params in list_params:
      params.max_it = np.inf
      params.gap_stopping = 0
      params.time_stopping = np.inf
      params.accelerated = True
      params.verbose = False

   # return
   assert(len(list_names) == len(list_params))
   assert(len(list_params) == nb_algs)

   return list_params, list_names