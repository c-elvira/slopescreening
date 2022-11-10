# -*- coding: utf-8 -*-
import numpy as np

from experiments.SIAM.setup import Setup
from experiments.SIAM.xp_1_balls import xpparams


def process(setup, log=True):
   
   # ---- load ----
   solutions_filename = f"results/xp_setup{setup.setup_id}.npz"
   solutions = np.load(solutions_filename, allow_pickle=True)

   screenings_filename = f"results/xp_setup{setup.setup_id}_screening.npz"
   screenings = np.load(screenings_filename, allow_pickle=True)

      # mat_results_gap[1, i_dic, i_seq, rep, i_ratio]
   mat_pvopt = solutions["mat_pvopt"]

   mat_nb_zero_detected = screenings['mat_nb_zero_detected']
   nb_test = mat_nb_zero_detected.shape[0]
   list_tests   = screenings["list_test"]
   list_legends = screenings["list_legends"]

   # ---- log ----
   if log:
      print("Experiment info")
      # print(f"- run with version {solutions["version"]}")
      print(f"- setup{setup.setup_id}")
      print("")

   # ---- processing ----
   mat_pc_detected =np.zeros(
      (nb_test, xpparams.nb_point, setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, setup.n_rep)
   )

   for i_dic in range(setup.nb_dic):
      for i_seq in range(setup.nb_sequence):
         for i_ratio in range(setup.nb_ratio_lbd):
            for rep in range(setup.n_rep):

               nb_0 = np.sum(mat_pvopt[i_dic, i_seq, i_ratio, rep, :] == 0)
               if nb_0 > 0:
                  mat_detected = mat_nb_zero_detected[:, :, i_dic, i_seq, i_ratio, rep]
                  mat_pc_detected[:, :, i_dic, i_seq, i_ratio, rep] = mat_detected / float(nb_0)


   # ---- return ----
   return {
      "mat_pc_detected": np.mean(mat_pc_detected, axis=5),
      "list_tests": list_tests,
      "list_legends": list_legends,
   }

   # {
   #    "results_mat_nb": results_mat_nb,
   #    "results_mat_nbz": results_mat_nbz,
   # }