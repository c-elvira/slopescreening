# -*- coding: utf-8 -*-
import numpy as np

from experiments.SIAM.setup import Setup


def process(folder, bool_it, setup, log=True):
   
   # ---- load ----
   results_file_name = f"{folder}setup{setup.setup_id}_"
   if bool_it:
      results_file_name += "d_results_it.npz"
   else:
      results_file_name += "c_results.npz"

   out = np.load(results_file_name, allow_pickle=True)

   if bool_it:
      mat_results = out['mat_results_it']
   else:
      # mat_results_gap[0, i_dic, i_seq, i_lbd, :]
      mat_results = out['mat_results']


   # ---- processing ----
   vec_tau = np.logspace(-16, 0, num=200, base=10.)

   # Fonction rho(tau) %xp such that gap <= tau 
   results_rho = np.full(
      (mat_results.shape[0], setup.nb_dic, setup.nb_sequence, setup.nb_ratio_lbd, vec_tau.size),
      np.nan
   )

   for i in range(vec_tau.size):
      results_rho[:, :, :, :, i] = np.mean(mat_results <= vec_tau[i], axis=4)


   # # Sparsity level
   # sparsity_level = 100 * np.mean(mat_results_nbnz, axis=3) / setup.n

   # # tim
   # results_average_it = np.mean(mat_results_it, axis=4)
   # results_std_it     = np.std(mat_results_it, axis=4)


   # ---- return ----
   return {
      "vec_tau": vec_tau,
      "results_rho": results_rho,
      # "sparsity_level": sparsity_level,
      # "results_average_it": results_average_it,
      # "results_std_it": results_std_it,
   }