# -*- coding: utf-8 -*-
import argparse, sys

import numpy as np

from slopescreening import __version__
from slopescreening.solver.slope import slope_gp
from slopescreening.solver.parameters import SlopeParameters, EnumLipchitzOptions
from slopescreening.utils import get_lambda_max, gamma_sequence_generator
from slopescreening.dictionaries import generate_dic

from experiments.SIAM.setup import Setup
from experiments.SIAM.slopepb import SlopePb

from get_algs_params import get_alg_params, get_nb_algs


parser=argparse.ArgumentParser()
parser.add_argument('--id', help='setup id', type=str, default="SIAM")
parser.add_argument('--erase', help='restart xp', action="store_true")
parser.add_argument('--precision', help='stop when gap reaches 1e-precision', default=8)
parser.add_argument('--exact', action="store_true")
parser.add_argument('--noverbose', help='disable printing if true', action="store_true")
args=parser.parse_args()


# -------------------------
#       Load  Setup
# -------------------------

setup = Setup(args.id)
folder = f'results/1e-{args.precision}'
folder += 'exact/' if args.exact else 'gersh/'

state_file_name   = f"{folder}setup{args.id}_a_state.npz"
time_file_name    = f"{folder}setup{args.id}_b_times.npz"
results_file_name = f"{folder}setup{args.id}_c_results.npz"


out_state = np.load(state_file_name, allow_pickle=True)
mat_seed  = out_state["mat_seed"]

out_times = np.load(time_file_name, allow_pickle=True)
mat_times = out_times["mat_times"]
# mat_it    = out_times["mat_it"]


mat_results = np.full(
   (get_nb_algs(setup), setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, setup.n_rep),
   np.nan
)

# stopping_gap = 10**(-float(args.precision))
update_lip   = EnumLipchitzOptions.EXACT if args.exact else EnumLipchitzOptions.GERSHGORIN

# --------------------------
#   Load  existing results
# --------------------------

try:
   if args.erase:
      raise FileNotFoundError

   load_results = np.load(results_file_name, allow_pickle=True)
   mat_results  = load_results['mat_results']

except FileNotFoundError:
   # Everything goes well
   pass


nb_xp = setup.nb_dic * setup.nb_sequence * setup.nb_ratio_lbd * setup.n_rep
t = 0
if not args.noverbose:
   print(f"Starting \"xp time\" with id {args.id}, precision 1e-{args.precision}")
for i_dic in range(setup.nb_dic):
   for i_seq in range(setup.nb_sequence):
      for i_ratio, ratio in enumerate(setup.list_ratio_lbd):

         stopping_time = np.median(mat_times[i_dic, i_seq, i_ratio, :])
         if not np.any(np.isnan(mat_results[:, i_dic, i_seq, i_ratio, :])):
            t += setup.n_rep
            continue

         for rep in range(setup.n_rep):
            if not args.noverbose:
               print(f"xp time {t+1} / {nb_xp}")

            # print(i_dic, i_seq, i_ratio, rep)
            # i_dic, i_seq, i_ratio, rep = 1, 0, 2, 17


            # --- set seed ---
            np.random.seed(mat_seed[i_dic, i_seq, i_ratio, rep])


            # --- Data and parameters ---
            matA = generate_dic(
               setup.list_dic[i_dic],
               setup.m,
               setup.n,
               setup.normalize
            )

            vecy  = np.random.randn(setup.m)
            vecy /= np.linalg.norm(vecy)

            vec_gammas = gamma_sequence_generator(
               setup.m, 
               setup.n,
               setup.list_sequence[i_seq][0],
               setup.list_sequence[i_seq][1:]
            )

            lbd_max = get_lambda_max(vecy, matA, vec_gammas)
            slopePb = SlopePb(matA, vecy, vec_gammas, ratio, lbdmax=lbd_max)

            # --- Solve slope problems ---
            list_params, _ = get_alg_params(setup, vec_gammas, args.exact)


            for i_alg, params in enumerate(list_params):

               if not np.isnan(mat_results[i_alg, i_dic, i_seq, i_ratio, rep]):
                  continue

               params.lipchitz_update = update_lip
               params.time_stopping   = stopping_time

               out_slope = slope_gp(vecy, matA, ratio * lbd_max, vec_gammas, params)

               best_gap = np.min(out_slope["gap"])

               vecx_hat = out_slope["sol"]
               vecu_hat = slopePb.make_dual_scaling(vecy - matA @ vecx_hat)
               gap      = slopePb.eval_gap(vecx_hat, vecu_hat)

               mat_results[i_alg, i_dic, i_seq, i_ratio, rep] = min(gap, best_gap)


            # --- Saving ---
            np.savez(results_file_name,
               mat_results     = mat_results,
               version       = __version__,
               allow_pickle  = True
            )

            t += 1
