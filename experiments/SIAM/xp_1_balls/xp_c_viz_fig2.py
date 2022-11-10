# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np

# XP import
from experiments.SIAM.slopepb import SlopePb
from experiments.SIAM.setup import Setup

import xpparams
from process_data import process


parser=argparse.ArgumentParser()
parser.add_argument('--noshow', help='do not display figures', action="store_true")
parser.add_argument('--noverbose', help='disable printing if true', action="store_true")
parser.add_argument('--save', help='save figure', action="store_true")
parser.add_argument('--id', help='setup id', type=str, default="SIAM")
args=parser.parse_args()

import matplotlib
if args.noshow:
   matplotlib.use("ps")
else:
   matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import matplotlib.font_manager as font_manager
# print(matplotlib.get_backend())
# matplotlib.rc('pdf', fonttype=42)

# -------------------------
#        Load Results
# -------------------------

# OSCAR
setup_oscar = Setup(args.id)
dic_process_oscar = process(setup_oscar)

mat_pc_detected_oscar = dic_process_oscar["mat_pc_detected"]
list_tests_oscar      = dic_process_oscar["list_tests"]


# -------------------------
#        Plot Results
# -------------------------

# vec_offsets = np.linspace(0, setup.max_offset, setup.nb_point)
# vec_offsets = setup_oscar.max_offset * np.logspace(-6., .0, num=setup_oscar.nb_point)


i_dic = 0
i_lbd = 1

fs=22
fs_ylabels = 20
list_colors = ["tab:orange", "tab:green", "tab:blue"]
list_style = ['-', '-.', '--']

i_artist = 0
class AnyObjectHandler(HandlerBase):
   def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):

      global i_artist
      l1 = plt.Line2D([x0,y0+width], [1.1*height,1.1*height], 
         linestyle=list_style[i_artist], color=list_colors[0],
         linewidth=4
      )
      l2 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], 
         linestyle=list_style[i_artist], color=list_colors[1],
         linewidth=4
      )
      l3 = plt.Line2D([x0,y0+width], [-.1*height,-.1*height], 
         linestyle=list_style[i_artist], color=list_colors[2],
         linewidth=4
      )
      i_artist += 1
 
      return [l1, l2, l3]


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
   fname='../fonts/ectt1000.ttf',
   weight='bold',
   style='normal',
   size=fs+6
)

f, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)


for i_seq, _ in enumerate(setup_oscar.list_sequence):

   ax[i_seq].set_title(
      f"OSCAR-{i_seq+1}",
      fontproperties=font_ttt,
   )
 
   for i_test, test_name in enumerate(list_tests_oscar):

      for tick in ax[i_seq].yaxis.get_major_ticks():
         tick.label.set_fontsize(18)

      
      for i_lbd in range(3):
         ax[i_seq].plot(
            xpparams.vec_offsets, 
            100 * mat_pc_detected_oscar[i_test, :, i_dic, i_seq, i_lbd],
            list_style[i_lbd],
            # label = "sphere p-test" if i_test==1 else "sphere single test",
            linewidth=4.,
            alpha=.9,
            color=list_colors[i_test]
         )

 
      # ax[0, i_seq].set_xlim([-.02, 2.02])  
      ax[i_seq].set_xscale("log")


ax[0].set_xlim([5e-5, 5e-1])
ax[0].set_ylim([-1, 101])
ax[0].set_ylabel(
   "% of zero entries detected", 
   fontproperties=font_math,
   fontsize=fs_ylabels+6,
)

ax[0].legend([object, object, object], ['$\\lambda / \\lambda_{\\max}=' + str(x) + '$' for x in [.3, .5, .8]],
           handler_map={object: AnyObjectHandler()},
           prop=font_math
)

f.tight_layout()




for i_seq in range(3):
   ax[i_seq].set_xlabel("$R_0$", fontproperties=font_math, fontsize=fs_ylabels+6)

   # ax[1, i_seq].xaxis.set_major_locator(plt.MaxNLocator(6))

   for tick in ax[i_seq].xaxis.get_major_ticks():
      tick.label.set_fontproperties(font_math)
      tick.label.set_fontsize(22)

   for tick in ax[i_seq].yaxis.get_major_ticks():
      tick.label.set_fontproperties(font_math)
      tick.label.set_fontsize(22)


if not args.noshow:
   plt.show()


if args.save:
   folderfig = "figs"
   Path(folderfig).mkdir(parents=True, exist_ok=True)

   filename = folderfig + f"/xp0_{setup_oscar.list_dic[i_dic]}.eps"
   # plt.rcParams['pdf.fonttype'] = 42
   plt.savefig(filename, bbox_inches='tight')
