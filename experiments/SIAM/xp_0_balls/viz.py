# -*- coding: utf-8 -*-
from decimal import Decimal
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Slope import
from src.utils import gamma_sequence_generator

# XP import
from xps.SIAM.slopepb import SlopePb
from xps.SIAM.setup import Setup
from xps.SIAM.xp_1_balls import xpparams
from xps.SIAM.xp_1_balls.process_data import process


parser=argparse.ArgumentParser()
parser.add_argument('--noshow', help='do not display figures', action="store_true")
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--id', help='setup id', type=str, default=1)
args=parser.parse_args()


# -------------------------
#        Load Results
# -------------------------

setup = Setup(args.id)
dic_process = process(setup)

mat_pc_detected = dic_process["mat_pc_detected"]
list_tests      = dic_process["list_tests"]
list_legends    = ["$r_q=1 \\forall q$", "Testing all $q$", "$r_q=q \\forall q$"]

# -------------------------
#        Plot Results
# -------------------------

for i_dic in range(setup.nb_dic):   

   if not np.any(mat_pc_detected[:, :, i_dic, :, :] > 0):
      continue

   f, ax = plt.subplots(setup.nb_sequence, setup.nb_ratio_lbd+1)
   f.suptitle(f"{setup.list_dic[i_dic]}")

   for i_seq, str_seq in enumerate(setup.list_sequence):
      
      for i_lbd in range(len(setup.list_ratio_lbd)): 

         if i_seq ==0:
            ax[i_seq, i_lbd].set_title(f"$\\lambda={setup.list_ratio_lbd[i_lbd]}$")

         for i_test in [0, 2, 1]:
            ax[i_seq, i_lbd].plot(
               xpparams.vec_offsets, 
               100 * mat_pc_detected[i_test, :, i_dic, i_seq, i_lbd],
               # '-',
               # color="blue",
               label= list_legends[i_test],
               # ,
               # linestyle=style
            )

         # if j == 0:
         #    ax[j, i_seq].set_title(f"{setup.list_sequence[i_seq]} -- lbd={setup.list_ratio_lbd[j]}")
         # else:
         #    ax[j, i_seq].set_title(f"lbd={setup.list_ratio_lbd[j]}")
         
         if i_lbd == 0 and i_seq == 0: 
            ax[i_seq, i_lbd].legend()

         ax[i_seq, i_lbd].set_xscale('log')

         # if i_seq >= 2:
         #    ax[i_seq, i_lbd].set_xlim([0, .5])
         # if i_seq >= 3:
         #    ax[i_seq, i_lbd].set_xlim([0, .05])

      vec_gammas = gamma_sequence_generator(
         setup.m, 
         setup.n,
         setup.list_sequence[i_seq][0],
         setup.list_sequence[i_seq][1:],
         log=True
      )

      ax[i_seq, len(setup.list_ratio_lbd)].plot(vec_gammas)   

   if args.save:
      filename = f"figs/setup{args.id}_{setup.list_dic[i_dic]}.pdf"
      plt.savefig(filename, bbox_inches='tight')

if not args.noshow:
   plt.show()