# -*- coding: utf-8 -*-
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Algorithm import
from src import __version__
from src.dictionaries import generate_dic
from src.utils import get_lambda_max, gamma_sequence_generator

# Screening
from src.screening.singletest import GapSphereSingleTest
from src.screening.gap_ptest import GAP_Ptest
from src.screening.gap_rqtest import GAP_RQtest
from src.screening.kappa_test import Kappa_test

# XP import
from xps.SIAM.slopepb import SlopePb
from xps.SIAM.setup import Setup
from xps.SIAM.xp_1_balls import xpparams


parser=argparse.ArgumentParser()
parser.add_argument('--id', help='setup id', type=str, default="1a")
args=parser.parse_args()

# -------------------------
#       Load  Setup
# -------------------------

setup = Setup(args.id)
solutions_filename = f"results/xp_setup{args.id}.npz"

try:
   # Try to load existing results
   load_results = np.load(solutions_filename, allow_pickle=True)
   mat_seed  = load_results["mat_seed"]
   mat_pvopt = load_results["mat_pvopt"]
   mat_dvopt = load_results["mat_dvopt"]

except FileNotFoundError:
   print("No result found")
   sys.exit(1)


# -------------------------
#        Screening Xp
# -------------------------

# percentage of detected zero
n_rep = 5
mat_nb_zero_detected = np.full(
   (3, xpparams.nb_point, setup.nb_sequence, n_rep),
   np.nan,
   dtype=int
)
poc_filename = f"results/poc{args.id}_screening.npz"

i_xp = 0

i_dic   = 2
i_ratio = 1
ratio   = .5
i_seq   = 1
seq     = "OSCAR-1"

# seq_r = 1+np.arange(setup.n, dtype=int)
seq_r   = 1 + np.arange(setup.n, dtype=int) // 5

#1 + (1 + np.arange(setup.n, dtype=int)) // 2

# for rep in range(setup.n_rep):
for rep in range(n_rep):

   np.random.seed(mat_seed[i_dic, i_seq, rep])

   # ---- 1. Gen data ----
   matA = generate_dic(
      setup.list_dic[i_dic],
      setup.m,
      setup.n,
      setup.normalize
   )

   vecy = np.random.randn(setup.m)
   vecy /= np.linalg.norm(vecy)


   # ---- 2. Compute parameters ----
   vec_gammas = gamma_sequence_generator(
      setup.m, 
      setup.n,
      setup.list_sequence[i_seq][0],
      setup.list_sequence[i_seq][1:]
   )

   lbd_max = get_lambda_max(vecy, matA, vec_gammas)

   # ---- 3. XP ----
   i_xp += 1
   print(f"xp {i_xp}")

   slopePb = SlopePb(matA, vecy, vec_gammas, ratio, lbdmax=lbd_max)
   vecx_hat = mat_pvopt[i_dic, i_seq, i_ratio, rep, :]
   vecu_hat = mat_dvopt[i_dic, i_seq, i_ratio, rep, :]
   gap = slopePb.eval_gap(vecx_hat, vecu_hat)
   Atu = matA.T @ vecu_hat

   # ---- 3b. Build thin safe ball ----
   rgap = np.sqrt(2 * gap)

   # ---- 3c. Testing sphere ---- 
   list_tests = [
      GapSphereSingleTest(),
      Kappa_test(vec_gammas, np.arange(setup.n, dtype=np.double)),
      GAP_RQtest(vec_gammas, seq_r),
      # GAP_Ptest(vec_gammas),
   ]


   for i_offset, offset in enumerate(xpparams.vec_offsets):
      for i_test, test in enumerate(list_tests):
         out = test.apply_test(np.abs(Atu), gap, ratio * lbd_max, vec_gammas, offset_radius=offset)
         mat_nb_zero_detected[i_test, i_offset, i_seq, rep] = np.sum(out)

f, ax = plt.subplots(1, 1)
for i_test in range(3):
   ax.plot(np.mean(mat_nb_zero_detected[i_test, :, i_seq, :n_rep], axis=1))

plt.show()

   # # Save
   # np.savez(
   #    poc_filename,
   #    mat_nb_zero_detected=mat_nb_zero_detected,
   #    list_test = [test.get_name() for test in list_tests],
   #    version = __version__,
   #    allow_pickle=True
   # )