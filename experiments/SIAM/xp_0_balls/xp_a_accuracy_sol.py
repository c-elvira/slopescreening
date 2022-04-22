# -*- coding: utf-8 -*-
import time, argparse, sys

import numpy as np

# Algorithm import
from src import __version__
from src.solver.parameters import SlopeParameters, EnumLipchitzOptions
from src.solver.slope import primal_func, dual_func, slope_gp

# Generative models
from src.utils import get_lambda_max, gamma_sequence_generator
from src.dictionaries import generate_dic

# Screening
from src.screening.gap_test_all import GapTestAll

# XP import
from experiments.SIAM.slopepb import SlopePb
from experiments.SIAM.setup import Setup
from experiments.SIAM.xp_1_balls import xpparams

parser=argparse.ArgumentParser()
parser.add_argument('--erase', help='erase existing results', action="store_true")
parser.add_argument('--id', help='setup id', type=str)
args=parser.parse_args()


# -------------------------
#       Load  Setup
# -------------------------

setup = Setup(args.id)
out_file_name = f"results/xp_setup{args.id}.npz"

mat_seed = np.random.randint(
   0, 2**32-1,
   size=(setup.nb_dic, setup.nb_sequence, setup.n_rep), 
)

mat_pvopt = np.zeros(
   (setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, setup.n_rep, setup.n)
)

mat_dvopt = np.zeros(
   (setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, setup.n_rep, setup.m)
)


try:
   if args.erase:
      raise FileNotFoundError

   # Try to load existing results
   load_results = np.load(out_file_name, allow_pickle=True)
   mat_seed  = load_results["mat_seed"]
   mat_pvopt = load_results["mat_pvopt"]
   mat_dvopt = load_results["mat_dvopt"]

except FileNotFoundError:
   pass


# -------------------------
#            Xp
# -------------------------

# For each trial
# 1. Compute high accuracy solution
# 2. Create ideal safe ball
# 3. increase radius

nb_xp = setup.nb_dic * setup.n_rep * setup.nb_sequence * setup.nb_ratio_lbd
i_xp = 0
for i_dic in range(setup.nb_dic):
   for i_seq, seq in enumerate(setup.list_sequence):
      for rep in range(setup.n_rep):

         np.random.seed(mat_seed[i_dic, i_seq, rep])

         # ---- 1. Gen data ----
         matA = generate_dic(
            setup.list_dic[i_dic],
            setup.m,
            setup.n,
            setup.normalize
         )

         lip = np.linalg.norm(matA, ord=2)**2
         vecy = np.random.randn(setup.m)
         vecy /= np.linalg.norm(vecy, 2)


         # ---- 2. Compute parameters ----
         vec_gammas = gamma_sequence_generator(
            setup.m, 
            setup.n,
            setup.list_sequence[i_seq][0],
            setup.list_sequence[i_seq][1:]
         )

         lbd_max = get_lambda_max(vecy, matA, vec_gammas)


         # ---- 3. XP ----
         for i_ratio, ratio in enumerate(setup.list_ratio_lbd):
            i_xp += 1

            slopePb = SlopePb(matA, vecy, vec_gammas, ratio, lbdmax=lbd_max)
            vecx_hat = mat_pvopt[i_dic, i_seq, i_ratio, rep, :]
            vecu_hat = mat_dvopt[i_dic, i_seq, i_ratio, rep, :]

            gap = slopePb.eval_gap(vecx_hat, vecu_hat)
            print(f"xp {i_xp} / {nb_xp} --- (gap={gap})")

            if (gap > xpparams.stopping_gap):
               gap_old = gap

               # ---- 3a. Find solution ----
               params = SlopeParameters()
               params.vecx_init = np.copy(vecx_hat)
               params.lipchitz_constant = lip
               params.lipchitz_update = EnumLipchitzOptions.EXACT
               params.max_it = 1e7
               params.gap_stopping = xpparams.stopping_gap
               params.time_stopping = np.inf
               params.screening1 = GapTestAll(vec_gammas)
               params.eval_gap   = True
               params.eval_gap_it = setup.eval_gap_it
               params.accelerated = False
               params.verbose = False
               
               out_slope = slope_gp(vecy, matA, ratio * lbd_max, vec_gammas, params)

               dv  = vecy - matA @  out_slope["sol"]
               dv = slopePb.make_dual_scaling(dv)
               gap = slopePb.eval_gap(out_slope["sol"], dv)

               if gap <= gap_old:
                  mat_pvopt[i_dic, i_seq, i_ratio, rep, :] = out_slope["sol"]
                  mat_dvopt[i_dic, i_seq, i_ratio, rep, :] = dv

                  # Save
                  np.savez(out_file_name,
                     mat_seed=mat_seed,
                     mat_pvopt=mat_pvopt,
                     mat_dvopt=mat_dvopt,
                     version = __version__,
                     allow_pickle=True
                  )

               # if i_xp ==3:
               #    exit()