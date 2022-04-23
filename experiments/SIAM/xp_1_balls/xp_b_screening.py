# -*- coding: utf-8 -*-
import argparse

import numpy as np

# Algorithm import
from src import __version__
from src.dictionaries import generate_dic
from src.utils import get_lambda_max, gamma_sequence_generator

# Screening
from src.screening.gap_test_p_1 import GapTestPequalOne
from src.screening.gap_test_p_q import GapTestPequalQ
from src.screening.gap_test_all import GapTestAll

# XP import
from experiments.SIAM.slopepb import SlopePb
from experiments.SIAM.setup import Setup
from experiments.SIAM.xp_1_balls import xpparams


parser=argparse.ArgumentParser()
parser.add_argument('--id', help='setup id', type=str, default=1)
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
NB_TEST = 3
mat_nb_zero_detected = np.full(
    (NB_TEST, xpparams.nb_point, setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, setup.n_rep),
    np.nan,
    dtype=int
)
screening_filename = f"results/xp_setup{args.id}_screening.npz"

i_xp = 0
nb_xp = setup.nb_dic * setup.n_rep * setup.nb_sequence * setup.nb_ratio_lbd
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
         for i_ratio, ratio in enumerate(setup.list_ratio_lbd):
            i_xp += 1
            print(f"xp {i_xp} / {nb_xp}")

            slopePb = SlopePb(matA, vecy, vec_gammas, ratio, lbdmax=lbd_max)
            vecx_hat = mat_pvopt[i_dic, i_seq, i_ratio, rep, :]
            vecu_hat = mat_dvopt[i_dic, i_seq, i_ratio, rep, :]
            gap = slopePb.eval_gap(vecx_hat, vecu_hat)
            Atu = matA.T @ vecu_hat

            # ---- 3b. Build thin safe ball ----
            rgap = np.sqrt(2 * gap)

            # ---- 3c. Testing sphere ---- 
            list_tests = [
               GapTestPequalOne(vec_gammas, np.arange(setup.n, dtype=np.double)),
               GapTestPequalQ(),
               GapTestAll(vec_gammas),
            ]
            assert(len(list_tests) == NB_TEST)

            for i_offset, offset in enumerate(xpparams.vec_offsets):
               for i_test, test in enumerate(list_tests):
                  out = test.apply_test(np.abs(Atu), gap, ratio * lbd_max, vec_gammas, offset_radius=offset)
                  mat_nb_zero_detected[i_test, i_offset, i_dic, i_seq, i_ratio, rep] = np.sum(out)


         # Save
         np.savez(
            screening_filename,
            mat_nb_zero_detected=mat_nb_zero_detected,
            list_test = [test.get_name() for test in list_tests],
            version = __version__,
            allow_pickle=True
         )