# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--id', help='setup id', type=str)
parser.add_argument('--precision', help='stop when gap reaches 1e-precision', default=8)
parser.add_argument('--exact', action="store_true")
parser.add_argument('--noshow', help='show plots', action="store_true")
parser.add_argument('--save', help='save figure', action="store_true")
args=parser.parse_args()

import matplotlib
if args.noshow:
   matplotlib.use('PS')
else:
   matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from experiments.SIAM.setup import Setup
from process_data import process
from get_algs_params import get_alg_params, get_nb_algs
from src.utils import gamma_sequence_generator
import matplotlib.font_manager as font_manager


fs = 20

font_math = font_manager.FontProperties(
   fname='../fonts/cmunrm.ttf',
   # weight='normal',
   # style='normal',
   math_fontfamily="cm",
   size=fs+2
)

font_text = font_manager.FontProperties(
   fname='../fonts/cmunrm.ttf',
   # weight='normal',
   # style='normal',
   math_fontfamily="cm",
   size=fs+2
)

font_ttt = font_manager.FontProperties(
   # fname='../fonts/ectt1000.ttf',
   fname='../fonts/cmuntt.ttf',
   weight='bold',
   style='normal',
   size=fs
)

# -------------------------
#       Load  results
# -------------------------

folder = f'results/1e-{args.precision}'
folder += 'exact/' if args.exact else 'gersh/'
   # check if folder exists, create it otherwise
Path(folder).mkdir(parents=True, exist_ok=True)

setup = Setup(args.id)
_, list_names = get_alg_params(setup, np.zeros(5), args.exact)
dic_process = process(folder, False, setup)

vec_tau     = dic_process["vec_tau"]
results_rho = dic_process["results_rho"]


# -------------------------
#       Plot  results
# -------------------------

f, ax = plt.subplots(setup.nb_sequence, setup.nb_dic, figsize=(16, 13.),
   sharex=True, sharey=True,
   gridspec_kw = {'wspace':.05, 'hspace':.05}
)
# f.suptitle(f"{setup.list_dic[i_dic]} dictionary", fontsize=14)

list_algnames = ["PG-no", "PGp=q", "PG-all"]
list_dicname  = ["Gaussian", "Uniform", "Toeplitz"]
i_lbd = 1

print("ploting time fig for")
print(f" - {list_dicname} dictionaries")
print(f" - lbd / lbd_max = {setup.list_ratio_lbd[i_lbd]}")

for i_dic in range(setup.nb_dic):   
   for i_seq in range(setup.nb_sequence):

      list_colors = ["tab:blue", "tab:orange", "tab:green"]
      for i_alg in range(len(list_names)):
         ax[i_seq, i_dic].plot(
            vec_tau, 
            100. * results_rho[i_alg, i_dic, i_seq, i_lbd],
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
            fontproperties=font_text
         )
            
      if i_dic == 2 and i_seq == 2:
         ax[i_seq, i_dic].legend(prop=font_ttt)

      if i_seq == setup.nb_sequence-1:
         ax[i_seq, i_dic].set_xlabel("$\\delta$ (Dual gap)", 
            # fontsize=14,
            fontproperties=font_text)

      if i_dic == 0:
         ax[i_seq, i_dic].set_ylabel(
            r"$\rho_{\ttsolv}(\delta)$",
            fontsize=fs+4,
            fontproperties=font_math
         )

         for tick in ax[i_seq, i_dic].yaxis.get_major_ticks():
            tick.label.set_fontproperties(font_math)
            tick.label.set_fontsize(16) 

      if i_seq == 2:
         for tick in ax[i_seq, i_dic].xaxis.get_major_ticks():
            tick.label.set_fontproperties(font_math)
            tick.label.set_fontsize(16)




if args.save:
   fig_name = f"figs/setup{args.id}_time"

   plt.savefig(fig_name + ".eps", bbox_inches='tight')

if not args.noshow:
   plt.show()
