# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path

import numpy as np

from src import __version__
from src.solver.slope import slope_gp
from src.solver.parameters import SlopeParameters, EnumLipchitzOptions
from src.utils import get_lambda_max, gamma_sequence_generator
from src.dictionaries import generate_dic
from src.screening.gap_ptest import GAP_Ptest

from experiments.SIAM.setup import Setup


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

state_file_name = f"{folder}setup{args.id}_a_state.npz"
time_file_name  = f"{folder}setup{args.id}_b_times.npz"

out = np.load(state_file_name, allow_pickle=True)
mat_seed = out["mat_seed"]

mat_times = np.full(
   (setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, setup.n_rep),
   np.nan
)

mat_it = np.full(
   (setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, setup.n_rep),
   np.nan, dtype=int
)

stopping_gap = 10**(-float(args.precision))
update_lip   = EnumLipchitzOptions.EXACT if args.exact else EnumLipchitzOptions.GERSHGORIN


# --------------------------
#   Load  existing results
# --------------------------

try:
   if args.erase:
      raise FileNotFoundError

   load_results = np.load(time_file_name, allow_pickle=True)
   mat_times  = load_results['mat_times']

except FileNotFoundError:
   # Everything goes well
   pass


nb_xp = setup.nb_dic * setup.nb_sequence * setup.nb_ratio_lbd * setup.n_rep
t = 0
print(f"Starting \"get budget\" with id {args.id}, precision 1e-{args.precision}")
for i_dic in range(setup.nb_dic):
   for i_seq in range(setup.nb_sequence):
      for i_ratio, ratio in enumerate(setup.list_ratio_lbd):
         for rep in range(setup.n_rep):
            print(f"xp budget {t+1} / {nb_xp}")

            if not np.isnan(mat_times[i_dic, i_seq, i_ratio, rep]):
               t += 1
               continue


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
            params = SlopeParameters()
            params.screening2 = GAP_Ptest(vec_gammas)
            params.screening_it_div = 2

            params.max_it        = np.inf
            params.gap_stopping  = stopping_gap
            params.time_stopping = np.inf
            params.lipchitz_update = update_lip
            params.accelerated = True
            params.verbose = False
            # max_eig = .np.linalg.norm(matA, 2)**2

            out_slope = slope_gp(vecy, matA, ratio * lbd_max, vec_gammas, params)

            mat_times[i_dic, i_seq, i_ratio, rep] = out_slope["time_run"]
            mat_it[i_dic, i_seq, i_ratio, rep]    = out_slope["nb_it"]


            # --- Saving ---
            np.savez(time_file_name,
               mat_times    = mat_times,
               mat_it       = mat_it,
               version      = __version__,
               allow_pickle = True
            )

            t += 1

