# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path

import numpy as np

from src import __version__
from src.solver.slope import slope_gp

from experiments.SIAM.setup import Setup
from get_algs_params import get_alg_params, get_nb_algs


# --------------------
#   Parse arguments
# --------------------

parser=argparse.ArgumentParser()
parser.add_argument('--id', help='setup id', type=str)
parser.add_argument('--erase', help='restart xp', action="store_true")
parser.add_argument('--precision', help='stop when gap reaches 1e-precision')
parser.add_argument('--exact', action="store_true")
args=parser.parse_args()

setup = Setup(args.id)
dims = (setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, setup.n_rep)

mat_seed = np.random.randint(0, 2**8, size=dims)

folder = f'results/1e-{args.precision}'
folder += 'exact/' if args.exact else 'gersh/'

   # check if folder exists, create it otherwise
Path(folder).mkdir(parents=True, exist_ok=True)

time_file_name = f"{folder}setup{args.id}_a_state.npz"

# --------------------
#        Xp
# --------------------

print(f"Starting \"get seed\" with id {args.id}, precision 1e-{args.precision}")
try:
   if args.erase:
      raise FileNotFoundError

   load_results = np.load(time_file_name, allow_pickle=True)

   if not args.erase:
      print("experiment already exists -- exit")
      sys.exit(1)

except FileNotFoundError:
   # Everything goes well
   pass

np.savez(
   time_file_name,
   mat_seed     = mat_seed,
   allow_pickle = True
)
