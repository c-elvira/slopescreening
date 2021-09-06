# -*- coding: utf-8 -*-
import time, argparse, sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from src import __version__
from src.screening import AbstractGapScreening
from src.slope import primal_func, dual_func
from src.utils import get_lambda_max, gamma_sequence_generator, compute_coherence, compute_coherence_function
from src.dictionaries import generate_dic

from setup import Setup


parser=argparse.ArgumentParser()
parser.add_argument('--noshow', help='save figure', action="store_true")
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--id', help='setup id', type=str,
    default=1)
args=parser.parse_args()


# -------------------------
#       Load  Setup
# -------------------------

setup = Setup(args.id)
out_file_name = f"results/xp_setup{args.id}.npz"

try:
   load_results = np.load(out_file_name, allow_pickle=True)
   mat_seed = load_results["mat_seed"]
   mat_sol = load_results["mat_sol"]


except FileNotFoundError:
   print("No experiment results found")
   exit()


# -------------------------
#     Screening class
# -------------------------

class DebugTest_p(AbstractGapScreening):
   """ Generalized test
   """
   def __init__(self, vec_cumsum_gammas):
      super(AbstractGapScreening, self).__init__()
      
      # Adding zero at the beginning
      self.vec_cumsum_gammas = np.zeros(vec_cumsum_gammas.size+1)
      self.vec_cumsum_gammas[1:] = vec_cumsum_gammas


   def apply_test(self, Atc, gap, lbd, vec_gammas, offset_radius=0, index=None) -> np.ndarray:
      n = Atc.size

      radius = np.sqrt(2 * gap) + offset_radius

         # 1. Sort in descenting order
      if index is None:
         index = np.argsort(np.abs(Atc))[::-1]

      for l in range(n):
         # setdiff1d with "assume_unique" option keeps the ordering
         cumsum_Atc = np.zeros(n)
         cumsum_Atc[1:] = np.cumsum(np.abs(Atc[np.setdiff1d(index, [l], assume_unique=True)]))

            # row: q
            # column: p
         map_accept = np.zeros((n, n))
         for q in range(1, n+1):
            range_p = np.arange(q)
            vec_bounds_p = np.abs(Atc[l]) \
               + (cumsum_Atc[q-1] - cumsum_Atc[q - 1 - range_p]) \
               + (1. + range_p) * radius \
               + lbd * self.vec_cumsum_gammas[:q][::-1]

               # screening test
            map_accept[q-1, :q] += (vec_bounds_p < lbd * self.vec_cumsum_gammas[q])

      return map_accept / float(setup.n)


# -------------------------
#            Hepers
# -------------------------

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


# -------------------------
#            Xps
# -------------------------

nb_xp = setup.nb_dic * setup.n_rep * setup.nb_sequence * setup.nb_ratio_lbd
i_xp = 0
for i_dic in range(setup.nb_dic):
   for i_seq, seq in enumerate(setup.list_sequence):
      for i_ratio, ratio in enumerate(setup.list_ratio_lbd):

         f, ax = plt.subplots(1, 1)
         mat_results = np.zeros((setup.n, setup.n))

         for rep in range(setup.n_rep):
            i_xp += 1
            print(f"xp {i_xp} / {nb_xp}")

            np.random.seed(mat_seed[i_dic, i_seq, rep])

            # ---- 1. Gen data ----
            matA = generate_dic(
               setup.list_dic[i_dic],
               setup.m,
               setup.n,
               setup.normalize
            )

            vecy = np.random.randn(setup.m)


            # ---- 2. Compute parameters ----
            vec_gammas = gamma_sequence_generator(
               setup.m, 
               setup.n,
               setup.list_sequence[i_seq],
               setup.m / (10 * setup.n)
            )

            lbd_max = get_lambda_max(vecy, matA, vec_gammas)


            # ---- 3. XP ----

            vecu, gap = compute_quantities(
               vecy, matA, mat_sol[i_dic, i_seq, i_ratio, rep, :], ratio * lbd_max, vec_gammas
            )

            # ---- 3c. Testing sphere ----
            test = DebugTest_p(np.cumsum(vec_gammas))

            Atu = matA.T @ vecu
            rgap = np.sqrt(2 * gap)

            # 4. Computing duality when taking into account the output of screening rules
            mat_results += test.apply_test(Atu, gap, ratio * lbd_max, vec_gammas, 0.) / setup.n_rep

         # [i_ratio, i_seq]
         heatmap = ax.pcolor(mat_results, cmap='viridis')
         # ax.imshow(mat_results, cmap='viridis')
         plt.colorbar(heatmap)
         ax.set_title(f"{setup.list_dic[i_dic]} -- {setup.list_sequence[i_seq]} -- lbd/lbd_max={ratio}")

         if not args.noshow:
            plt.draw()
            plt.pause(0.001)

         if args.save:
            filename = f"figs/viztestp/setup{args.id}_{setup.list_dic[i_dic]}_seq{i_seq}_lbd{i_ratio}.pdf"
            plt.savefig(filename, bbox_inches='tight')

if not args.noshow:
   plt.show()
