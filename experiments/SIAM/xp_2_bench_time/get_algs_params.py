import numpy as np

from src.solver.parameters import SlopeParameters, EnumLipchitzOptions, DualScalingOptions

from src.screening.gap_test_p_q import GapTestPequalQ
from src.screening.gap_test_all import GapTestAll


nb_algs = 4

def get_nb_algs(setup):

   return nb_algs


def get_alg_params(setup, vec_gammas, exact):

   list_params = []
   list_names  = []

   # --- 1. Xp ---

      # -- 1a. No screening
   params1 = SlopeParameters()
   params1.eval_gap = False
   list_params.append(params1)
   list_names.append("no screening")


      # -- 1b. Strategy p_q=q
   params2 = SlopeParameters()
   params2.screening1 = GapTestPequalQ()
   params2.eval_gap_it = setup.eval_gap_it
   list_params.append(params2)
   list_names.append("p_q=q")


      # -- 1c. Strategy all
   params3 = SlopeParameters()
   params3.screening1 = GapTestAll(vec_gammas)
   params3.eval_gap_it = setup.eval_gap_it
   list_params.append(params3)
   list_names.append("test-all")


   # -- 1d. Bao et al.
   params4 = SlopeParameters()
   params4.screening1 = GapTestPequalQ()
   params4.dual_scaling = DualScalingOptions.BAO_ET_AL
   params4.eval_gap_it = setup.eval_gap_it
   list_params.append(params4)
   list_names.append("Bao-et-al")


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