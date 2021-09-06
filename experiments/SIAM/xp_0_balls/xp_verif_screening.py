# -*- coding: utf-8 -*-
import time, argparse, sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from src import __version__
from src.parameters import SlopeParameters, EnumLipchitzOptions
from src.slope import primal_func, dual_func, slope_gp
from src.screening import SafeGapSphere, SafeGapSphereSingleTest, GeneralizedGapSphere, GeneralizedGapSphereV2, GeneralizedGapSphereV3
from src.utils import get_lambda_max, gamma_sequence_generator, compute_coherence, compute_coherence_function
from src.dictionaries import generate_dic

from setup import Setup


parser=argparse.ArgumentParser()
parser.add_argument('--erase', help='save figure', action="store_true")
parser.add_argument('--continue', help='save figure', action="store_true", dest="cont")
parser.add_argument('--id', help='setup id', type=str,
    default=1)
args=parser.parse_args()


# -------------------------
#       Load  Setup
# -------------------------

setup = Setup(args.id)
out_file_name = f"results/xp_setup{args.id}.npz"

try:
   if args.erase:
      raise FileNotFoundError

   # Try to load existing results
   load_results = np.load(out_file_name, allow_pickle=True)
   mat_seed = load_results["mat_seed"]
   mat_sol = load_results["mat_sol"]

   date_start_xp = load_results['date_start_xp']

except FileNotFoundError:

   pass

# -------------------------
#            Xp
# -------------------------

# For each trial
# 1. Compute high accuracy solution
# 2. Create ideal safe ball
# 3. increse radius

   # When does the xp have started?
print("#"*50)
if type(date_start_xp) == np.ndarray:
   print("#\tdate and time =", date_start_xp)
else:
   dt_string = date_start_xp.strftime("%d/%m/%Y %H:%M:%S")
   print("#\tdate and time =", dt_string)
print(f"#\tXp setup{args.id}")
print("#"*50)
print("")

def compute_quantities(vecy, matA, vecx, lbd, vec_gammas):
   """
   """

   # residual error
   vecu = vecy - matA @ vecx

      # dual scaling
   beta_dual = np.sort(np.abs(matA.T @ vecu))[::-1]
   beta_dual = np.cumsum(beta_dual) / np.cumsum(lbd * vec_gammas)
   vecu /= np.max(beta_dual)

   pval = primal_func(vecy, matA @ vecx, vecx, lbd, vec_gammas)
   dval = dual_func(vecy, np.linalg.norm(vecy, 2)**2, vecu)

   gap = np.abs(pval - dval)

   return vecu, gap


nb_xp = setup.nb_dic * setup.n_rep * setup.nb_sequence * setup.nb_ratio_lbd
i_xp = 0
for i_dic in range(setup.nb_dic):
   for i_seq, seq in enumerate(setup.list_sequence):
      for rep in range(setup.n_rep):
         # if not np.any(np.isnan(mat_nb[1, :, i_dic, i_seq, :, rep])):
         #    continue

         np.random.seed(mat_seed[i_dic, i_seq, rep])

         # ---- 1. Gen data ----
         matA = generate_dic(
            setup.list_dic[i_dic],
            setup.m,
            setup.n,
            setup.normalize
         )

            # Abs is to protect from complex numbers
         max_eig = np.linalg.norm(matA, ord=2)**2

         coherence          = compute_coherence(matA)
         coherence_function = compute_coherence_function(matA)

         vecy = np.random.randn(setup.m)


         # ---- 2. Compute parameters ----
         vec_gammas =gamma_sequence_generator(
            setup.m, 
            setup.n,
            setup.list_sequence[i_seq],
            setup.m / (10 * setup.n)
         )

         lbd_max = get_lambda_max(vecy, matA, vec_gammas)


         # ---- 3. XP ----
         for i_ratio, ratio in enumerate(setup.list_ratio_lbd):
            i_xp += 1
            print(f"xp {i_xp} / {nb_xp}")


            vecu, gap = compute_quantities(
               vecy, matA, mat_sol[i_dic, i_seq, i_ratio, rep, :], ratio * lbd_max, vec_gammas
            )

            # if (gap_old <= setup.stopping_gap) and not np.any(np.isnan(mat_nb[:, :, i_dic, i_seq, i_ratio, rep])):
               # continue
            # else:
            if (gap > setup.stopping_gap):
               gap_old = gap

               # ---- 3a. Find solution ----
               params = SlopeParameters()
               params.vecx_init = mat_sol[i_dic, i_seq, i_ratio, rep, :]
               params.lipchitz_constant = max_eig
               params.lipchitz_update = EnumLipchitzOptions.GERSHGORIN
               params.max_it = 1e10
               params.gap_stopping = setup.stopping_gap
               params.time_stopping = np.inf
               params.screening1 = SafeGapSphereSingleTest()
               params.screening2 = SafeGapSphere(np.cumsum(vec_gammas), coherence_function)
               params.screening_it_div = 1.
               params.accelerated = True
               params.verbose = False
               
               out_slope = slope_gp(vecy, matA, ratio * lbd_max, vec_gammas, params)

               vecu, gap = compute_quantities(
                  vecy, matA, out_slope["sol"], ratio * lbd_max, vec_gammas
               )

               if gap <= gap_old:
                  mat_sol[i_dic, i_seq, i_ratio, rep, :] = out_slope["sol"]


            # ---- 3c. Testing sphere ----
            test1  = GeneralizedGapSphere(np.cumsum(vec_gammas))
            test2 = GeneralizedGapSphereV2(np.cumsum(vec_gammas))
            test3 = GeneralizedGapSphereV3(np.cumsum(vec_gammas))
            # test2 = GeneralizedCoherenceGapSphere(np.cumsum(vec_gammas), coherence_function)

            Atu = matA.T @ vecu
            rgap = np.sqrt(2 * gap)

            # 4. Computing duality when taking into account the output of screening rules
            print(f"  old gap   {gap}")
            delta = [0, 0, 0]
            sumzero = [0, 0, 0]
            for i_test, test in enumerate([test1, test2, test3]):
               t_1 = time.time()
               out = test.apply_test(Atu, gap, ratio * lbd_max, vec_gammas, 0.)
               delta[i_test] = time.time() - t_1

               sumzero[i_test] = np.sum(out)

               _, new_gap = compute_quantities(
                  vecy, matA[:, np.invert(out)], 
                  mat_sol[i_dic, i_seq, i_ratio, rep, np.invert(out)], 
                  ratio * lbd_max, vec_gammas[:(setup.n - np.sum(out))]
               )

               print(f"  new gap {i_test} {new_gap}")
            print(f"  delta time {delta[0] / delta[1]}")
            print(f"  delta time {delta[0] / delta[2]}")
            print(f"  nb zero {sumzero[0]} - {sumzero[1]} - {sumzero[2]}")
