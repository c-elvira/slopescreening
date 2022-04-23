# -*- coding: utf-8 -*-
import argparse

import numpy as np

# XP import
from experiments.SIAM.slopepb import SlopePb
from experiments.SIAM.setup import Setup

import xpparams
from process_data import process


parser=argparse.ArgumentParser()
parser.add_argument('--noshow', help='do not display figures', action="store_true")
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--id', help='setup id', type=str, default="SIAM")
parser.add_argument('--i_seq', type=int, default=0)
args=parser.parse_args()

import matplotlib
if args.noshow:
   matplotlib.use('PS')
else:
   matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import matplotlib.font_manager as font_manager

# -------------------------
#        Font stuff 
# -------------------------

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
   fname='../fonts/computer-modern/cmuntt.ttf',
   weight='bold',
   style='normal',
   size=fs
)


# -------------------------
#        Load Results
# -------------------------


setup_oscar = Setup(args.id)
dic_process_oscar = process(setup_oscar)

mat_pc_detected_oscar = dic_process_oscar["mat_pc_detected"]
list_tests_oscar      = dic_process_oscar["list_tests"]



# -------------------------
#        Plot Results
# -------------------------

i_dic = 2
i_lbd = 1

fs=22
fs_ylabels = 20
list_colors  = ["tab:blue", "tab:orange", "tab:green"]
list_legends = ["$p_q=q\\;\\forall q$", "all", "$p_q=1\\;\\forall q$"]
# "best $r_q \\;\\forall q$ "

print("printing xp_0_ball parameters with")
print(f"- {setup_oscar.list_dic[i_dic]} dictionary")
print(f"- lbd / lbd_max = {setup_oscar.list_ratio_lbd[i_lbd]}")


f, ax = plt.subplots(1, 1, figsize=(.7*16, .6*9), sharex=True, sharey=True)

# ax.set_title(f"OSCAR-{i_seq+1}", fontsize=fs+2)
ax.set_xlabel(
   r"$R$", 
   fontsize=fs+2,
   fontproperties=font_math,
)
ax.set_ylabel(
   "% of zero entries detected",
   fontsize=fs+2,
   fontproperties=font_text
)

for tick in ax.xaxis.get_major_ticks():
   tick.label.set_fontproperties(font_math)
   tick.label.set_fontsize(20)

for tick in ax.yaxis.get_major_ticks():
   tick.label.set_fontproperties(font_math)
   tick.label.set_fontsize(20)

for i_test in [2, 0, 1]:
   ax.plot(
      xpparams.vec_offsets, 
      100 * mat_pc_detected_oscar[i_test, :, i_dic, args.i_seq, i_lbd],
      label = list_legends[i_test],
      linewidth=4.,
      alpha=.9,
      color=list_colors[i_test]
   )

ax.legend(
   fontsize=fs-2,
   prop=font_math
)

ax.set_xscale("log")
ax.set_xlim([1e-6, 1e0])
ax.set_ylim([-2, 102])

if args.save:
   filename = f"figs/xp_illustration_screening{args.i_seq}.eps"
   plt.savefig(filename, bbox_inches='tight')

if not args.noshow:
   plt.show()