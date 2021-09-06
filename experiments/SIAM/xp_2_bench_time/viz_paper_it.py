# -*- coding: utf-8 -*-
import argparse

import numpy as np
import matplotlib.pyplot as plt

from xps.SIAM.setup import Setup
from process_data import process
from get_algs_params import get_alg_params, get_nb_algs
from src.utils import gamma_sequence_generator


parser=argparse.ArgumentParser()
parser.add_argument('--id', help='setup id', type=str)
parser.add_argument('--precision', help='stop when gap reaches 1e-precision', default=8)
parser.add_argument('--exact', action="store_true")
parser.add_argument('--noshow', help='show plots', action="store_true")
parser.add_argument('--save', help='save figure', action="store_true")
args=parser.parse_args()

# -------------------------
#       Load  results
# -------------------------

folder = f'results/1e-{args.precision}'
folder += 'exact/' if args.exact else 'gersh/'

setup = Setup(args.id)
_, list_names = get_alg_params(setup, np.zeros(5), args.exact)
dic_process = process(folder, True, setup)

vec_tau            = dic_process["vec_tau"]
results_rho        = dic_process["results_rho"]


# -------------------------
#       Plot  results
# -------------------------

f, ax = plt.subplots(1, setup.nb_dic, figsize=(16, 4.),
   sharex=True, sharey=True,
   gridspec_kw = {'wspace':.05, 'hspace':.05}
)

list_algnames = ["PG", "PG$_s$", "PG$_p$"]
list_dicname  = ["Gaussian", "Uniform", "toeplitz"]
i_lbd = 1
i_dic = 0

print("ploting it fig for")
print(f" - {list_dicname[i_dic]} dictionary")
print(f" - lbd / lbd_max = {setup.list_ratio_lbd[i_lbd]}")

for i_seq in range(setup.nb_sequence):

   list_colors = ["tab:blue", "tab:orange", "tab:green"]
   for i_alg in range(len(list_names)):

      ax[i_seq].plot(
         vec_tau, 
         100. * results_rho[i_alg, i_dic, i_seq, i_lbd],
         linewidth=3.,
         label=list_algnames[i_alg],
         color=list_colors[i_alg],
         linestyle='-.'
      )

   ax[i_seq].set_title(f"OSCAR-{i_seq+1}", fontsize=16)

   ax[i_seq].set_xscale('log')

   ax[i_seq].set_xlim([5e-15, 5e-5])
   ax[i_seq].set_ylim([-2, 101])
   ax[i_seq].grid()

   if i_seq == 0 :
      ax[i_seq].legend()   

   ax[i_seq].set_xlabel("$\\tau$ (Dual gap)", fontsize=14)

   if i_seq == 0:
      ax[i_seq].set_ylabel("$\\rho_s(\\tau)$", fontsize=14)



if args.save:
   fig_name = f"figs/setup{args.id}_it"


   plt.savefig(fig_name + ".pdf", bbox_inches='tight')

if not args.noshow:
   plt.show()
