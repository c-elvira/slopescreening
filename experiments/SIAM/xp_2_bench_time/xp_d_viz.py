# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np


parser=argparse.ArgumentParser()
parser.add_argument('--id', help='setup id', type=str, default="SIAM")
parser.add_argument('--precision', help='stop when gap reaches 1e-precision', default=8)
parser.add_argument('--exact', action="store_true")
# parser.add_argument('--it', help='show it results', action="store_true")
parser.add_argument('--ilbd', help='save figure', type=int, default=0)
parser.add_argument('--noshow', help='show plots', action="store_true")
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--noverbose', help='disable printing if true', action="store_true")
args=parser.parse_args()

import matplotlib
if args.noshow:
   matplotlib.use('PS')
else:
   matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True



from process_data import process
from get_algs_params import get_alg_params, get_nb_algs
from slopescreening.utils import gamma_sequence_generator
from experiments.SIAM.setup import Setup


fs = 20

# -------------------------
#       Load  results
# -------------------------

folder = f'results/1e-{args.precision}'
folder += 'exact/' if args.exact else 'gersh/'

setup = Setup(args.id)
_, list_names = get_alg_params(setup, np.zeros(5), args.exact)
dic_process = process(folder, False, setup)

vec_tau            = dic_process["vec_tau"]
results_rho        = dic_process["results_rho"]
# sparsity_level     = dic_process["sparsity_level"]
# results_average_it = dic_process["results_average_it"]
# results_std_it     = dic_process["results_std_it"]


# -------------------------
#       Plot  results
# -------------------------

f, ax = plt.subplots(setup.nb_sequence, setup.nb_dic, figsize=(16, 13.),
   sharex=True, sharey=True,
   gridspec_kw = {'wspace':.05, 'hspace':.05}
)
# f.suptitle(f"{setup.list_dic[i_dic]} dictionary", fontsize=14)

list_algnames = ["PG-no", "PG-p=q", "PG-all", 'PG-Bao']
list_dicname  = ["Gaussian", "Uniform", "Toeplitz"]
# args.ilbd = 1

if not args.noverbose:
   print("ploting time fig for")
   print(f" - {list_dicname} dictionaries")
   print(f" - lbd / lbd_max = {setup.list_ratio_lbd[args.ilbd]}")

for i_dic in range(setup.nb_dic):   
   for i_seq in range(setup.nb_sequence):

      list_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
      for i_alg in [0, 1, 3, 2]:#range(len(list_names)):
         ax[i_seq, i_dic].plot(
            vec_tau, 
            100. * results_rho[i_alg, i_dic, i_seq, args.ilbd],
            linewidth=3.,
            label=list_algnames[i_alg],
            color=list_colors[i_alg]
         )


      ax[i_seq, i_dic].set_xscale('log')

      ax[i_seq, i_dic].set_xlim([5e-15, 5e-1])
      ax[i_seq, i_dic].set_ylim([-2, 101])
      ax[i_seq, i_dic].grid()

      if i_seq == 0:
         ax[i_seq, i_dic].set_title(
            f"{list_dicname[i_dic]}",
            fontsize=fs+2,
         )
            
      if i_dic == 2 and i_seq == 2:
         ax[i_seq, i_dic].legend()

      if i_seq == setup.nb_sequence-1:
         ax[i_seq, i_dic].set_xlabel("$\delta$ (Dual gap)", 
            # fontsize=14,
         )

      if i_dic == 0:
         ax[i_seq, i_dic].set_ylabel(
            r"$\rho(\delta)$",
            fontsize=fs+4,
         )

         for tick in ax[i_seq, i_dic].yaxis.get_major_ticks():
            tick.label.set_fontsize(16) 

      if i_seq == 2:
         for tick in ax[i_seq, i_dic].xaxis.get_major_ticks():
            tick.label.set_fontsize(16)




if args.save:
   Path("figs").mkdir(parents=True, exist_ok=True)
   fig_name = f"figs/setup{args.id}_precision{args.precision}_exact{args.exact}_time"

   plt.savefig(fig_name + ".eps", bbox_inches='tight')

if not args.noshow:
   plt.show()
