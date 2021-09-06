# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path

import numpy as np

from src import __version__
from src.solver.slope import slope_gp
from src.solver.parameters import SlopeParameters, EnumLipchitzOptions
from src.screening.gap_ptest import GAP_Ptest
from src.utils import get_lambda_max, gamma_sequence_generator
from src.dictionaries import generate_dic

from experiments.SIAM.setup import Setup

from get_algs_params import get_alg_params, get_nb_algs


parser=argparse.ArgumentParser()
parser.add_argument('--id', help='setup id', type=str, default=1)
parser.add_argument('--erase', help='restart xp', action="store_true")
parser.add_argument('--precision', help='stop when gap reaches 1e-precision')
parser.add_argument('--exact', action="store_true")
args=parser.parse_args()


# -------------------------
#       Load  Setup
# -------------------------

setup = Setup(args.id)
folder = f'results/1e-{args.precision}'
folder += 'exact/' if args.exact else 'gersh/'
   # check if folder exists, create it otherwise
Path(folder).mkdir(parents=True, exist_ok=True)

state_file_name   = f"{folder}setup{args.id}_a_state.npz"
time_file_name    = f"{folder}setup{args.id}_b_times.npz"
results_file_name = f"{folder}setup{args.id}_c_results.npz"


out_state = np.load(state_file_name, allow_pickle=True)
mat_seed  = out_state["mat_seed"]

out_times = np.load(time_file_name, allow_pickle=True)
mat_times = out_times["mat_times"]


mat_results = np.full(
   (get_nb_algs(setup), setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, setup.n_rep),
   np.nan
)

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
print(f"Starting \"xp time\" with id {args.id}, precision 1e-{args.precision}")
for i_dic in range(setup.nb_dic):
   for i_seq in range(setup.nb_sequence):
      for i_ratio, ratio in enumerate(setup.list_ratio_lbd):

         stopping_time = np.mean(mat_times[i_dic, i_seq, i_ratio, :])
         if not np.any(np.isnan(mat_results[:, i_dic, i_seq, i_ratio, :])):
            t += setup.n_rep
            continue

         for rep in range(setup.n_rep):
            print(f"xp time {t+1} / {nb_xp}")


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

            # --- Solve slope problems ---
            list_params, _ = get_alg_params(setup, vec_gammas, args.exact)


            for i_alg, params in enumerate(list_params):
               if not np.isnan(mat_results[i_alg, i_dic, i_seq, i_ratio, rep]):
                  continue

               params.lipchitz_update = update_lip
               params.time_stopping   = stopping_time
               out_slope = slope_gp(vecy, matA, ratio * lbd_max, vec_gammas, params)

               mat_results[i_alg, i_dic, i_seq, i_ratio, rep] = out_slope["gap"]


            # --- Saving ---
            np.savez(results_file_name,
               mat_results     = mat_results,
               version       = __version__,
               allow_pickle  = True
            )

            t += 1
