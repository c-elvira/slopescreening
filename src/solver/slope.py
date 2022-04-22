# -*- coding: utf-8 -*-
import time

import numpy as np

from src.parameters import SlopeParameters, EnumLipchitzOptions, DualScalingOptions
from src.prox import prox_owl


def primal_func(vecy, Ax, x, lbd, vec_gammas) -> float:
   """
   """
   return .5 * np.linalg.norm(vecy - Ax)**2 \
      + lbd * vec_gammas.T @ (np.sort(np.abs(x))[::-1])


def dual_func(vecy, normy2, u) -> float:
   """
   """
   return .5 * (normy2 - np.linalg.norm(vecy - u, 2)**2)


def slope_gp(vecy, matA, lbd, vecgamma, algParameters=SlopeParameters()):
   """Proximal gradient algorithm to solve the SLOPE problem
   
   ::math
   
      \\arg\\min_{} \\dots 
   
   Parameters
   ----------
   vecy : numpy.array
      Observation vector
      size [m,]
   matA : numpy.array
      Dictionary matrix
      size [m, n]
   lbd : positive float
      regularization parameter
   vecgamma : numpy.array
      Slope parameters
      size [n,]
   algParameters : SlopeParameters
      Description

   Returns
   -------
   results : dict
      Python dictionary. keys
      * "sol": np.ndarray: Aproximated solution of the Slope problem
      * "cost_function": np.ndarray, cost function accross iterations
      * "dual_function": np.ndarray, dual cost function accross iterations
      * "gap": float, duality gap at convergence
      * "time_run": float running time
      * "nb_it" : int, number of iterations before stopping

   """

   # -------------------------
   #        Beginning
   # -------------------------

   __assert( **locals() )
   m, n = matA.shape
   ind_active = np.arange(n)
   n_active = n

   if not np.isfortran(matA):
      print("Warning: code runs faster if matA is in fortan order")


   # -------------------------
   #  Quantities of interest
   # -------------------------

      # Estimated solution
   vecx_hat = np.zeros(n) \
      if algParameters.vecx_init is None \
      else algParameters.vecx_init

      # Quantities for accelerated algorithm
   x_tilde = np.copy(vecx_hat) if algParameters.accelerated else None
   beta_k = 1.

      # Precomputed quantities
   Aty = matA.T @ vecy
   normy2 = np.linalg.norm(vecy)**2
   gamma_returned = vecgamma[::-1]

      # Coherence
   mu = algParameters.coherence

      # Coherence function
   vec_coherence_function = algParameters.coherence_function

      # Lipchitz constant
   lip = algParameters.lipchitz_constant
   lip = lip if lip is not None else np.inf


   def update_Lip():
      if algParameters.lipchitz_update == EnumLipchitzOptions.EXACT:

         if n_active <= 1:
            updt = 1.
         else:
            updt = np.linalg.norm(matA[:, ind_active], ord=2)**2

      elif algParameters.lipchitz_update == EnumLipchitzOptions.GERSHGORIN:
         updt = 1. + float(n_active - 1.) * mu

      elif algParameters.lipchitz_update == EnumLipchitzOptions.BACKTRACKING:
         raise NotImplemented

      else:
         raise "Lipchitz update rule recognized"

      return min(lip, updt)


   lip = update_Lip()


   # for screening purposes (if necessary)
   screening_test_1 = algParameters.screening1
   screening_test_2 = algParameters.screening2

   vec_cumsum_gammas = np.cumsum(vecgamma)

   is_test_1 = False if screening_test_1 is None else True
   is_test_2 = False if screening_test_2 is None else True

   gap_last_test = np.inf
   if is_test_2:
      do_test2 = lambda g, g_last: g <= g_last / algParameters.screening_it_div
      gap_tresh = algParameters.screening_it_div
      # do_test2 = lambda g, g_last: g <= gap_tresh and g_last > gap_tresh
   else:
      do_test2 = lambda g, g_last: False

   # -------------------------
   #          Loop
   # -------------------------

   vec_cost_func = []
   vec_dual_func = []
   vec_n_active  = []

   best_costfunc = .5 * normy2
   best_vecx     = np.copy(vecx_hat)
   best_dualfunc = 0

   it = 0

   # Security to prevent error with negative gap
   gap_stop = max(algParameters.gap_stopping, 5e-16)

   time_starting = time.time()
   ellapsed_time = lambda: time.time() - time_starting
   while True:

      # ------------------
      #  Common quantites
      # ------------------
      Ax = matA[:, ind_active] @ vecx_hat[ind_active]
      vec_res = vecy - Ax
      neg_grad = matA[:, ind_active].T @ vec_res

      # -- Evaluating primal function
      # I drop the idea below. it is the "best strategy" only when evaluating the dual function
      # ytAx = vecy.dot(Ax)
      # normAx2 = np.linalg.norm(Ax, 2)**2
      half_norm_res_2 = .5 * np.linalg.norm(vec_res, 2)**2
      lbd_slope_norm = lbd * gamma_returned.dot(np.sort(np.abs(vecx_hat[ind_active])))

      vec_cost_func.append(half_norm_res_2 + lbd_slope_norm)

      if vec_cost_func[-1] < best_costfunc:
         best_costfunc = vec_cost_func[-1]
         best_vecx = np.copy(vecx_hat)

      gap = best_costfunc


      if (it % 20 == 0) or algParameters.eval_gap:
         # -- Evaluating dual function (if needed)
         # if algParameters.eval_gap:
         index_sort_neg_grad = np.argsort(np.abs(neg_grad))[::-1]
         # print(index_sort_neg_grad)

         # Dual scaling
         if algParameters.dual_scaling == DualScalingOptions.EXACT:
            beta_dual = np.cumsum(np.abs(neg_grad[index_sort_neg_grad])) \
               / vec_cumsum_gammas[:n_active]
            coeff_dual_scaling = max(np.max(beta_dual) / lbd, 1.)

         elif algParameters.dual_scaling == DualScalingOptions.BAO_ET_AL:
            beta_dual = np.abs(neg_grad[index_sort_neg_grad]) / vecgamma[:n_active]
            coeff_dual_scaling = max(np.max(beta_dual) / lbd, 1.)

         else:
            raise NotImplemented

         yT_res = vecy.dot(vec_res)
         d_func_val = (yT_res - half_norm_res_2 / coeff_dual_scaling) / coeff_dual_scaling

         if algParameters.save_dfunc:
         	vec_dual_func.append(d_func_val)
         
         gap = np.abs(best_costfunc - d_func_val)


      # ------------------
      #   Stopping rules
      # ------------------

      if (it >= algParameters.max_it) or (gap <= gap_stop) or (ellapsed_time() > algParameters.time_stopping):
         break


      # ------------------
      #      Screening
      # ------------------

      if (it % 20 == 0) and (is_test_1 or is_test_2):
         # -- Evaluating dual function (if needed)
         # if algParameters.eval_gap:
         index_sort_neg_grad = np.argsort(np.abs(neg_grad))[::-1]
         # print(index_sort_neg_grad)

         # Dual scaling
         if algParameters.dual_scaling == DualScalingOptions.EXACT:
            beta_dual = np.cumsum(np.abs(neg_grad[index_sort_neg_grad])) \
               / vec_cumsum_gammas[:n_active]
            coeff_dual_scaling = max(np.max(beta_dual) / lbd, 1.)

         elif algParameters.dual_scaling == DualScalingOptions.BAO_ET_AL:
            beta_dual = np.abs(neg_grad[index_sort_neg_grad]) / vecgamma[:n_active]
            coeff_dual_scaling = max(np.max(beta_dual) / lbd, 1.)

         else:
            raise NotImplemented

         yT_res = vecy.dot(vec_res)
         d_func_val = (yT_res - half_norm_res_2 / coeff_dual_scaling) / coeff_dual_scaling

         # vec_dual_func.append(dualfunc)
         
         # we take vec_dual_func[-1] because the corresponding u is used for 
         # the GAP ball
         # gap = np.abs(best_costfunc - vec_dual_func[-1])
         gap = np.abs(best_costfunc - d_func_val)

         # Potential improvements
         # - l218: Atu coulb commented 
         # - l225: Atu coulb replaced by neg_grad and lbd by "coeff_dual_scaling * lbd" (+ radius)
         # - l242: idem
         # Atu = neg_grad / coeff_dual_scaling
         # gap = np.abs(best_costfunc - vec_dual_func[-1])

         # 1. Apply test
         if is_test_2: # and do_test2(gap, gap_last_test):
            out_test2 = screening_test_2.apply_test(np.abs(neg_grad), gap, lbd, vecgamma[:n_active], coeff_dual_scaling=coeff_dual_scaling, index=index_sort_neg_grad)

            if np.any(out_test2):
               # 2a. Set entry to 0
               vecx_hat[ind_active[out_test2]] = 0.

               # 2b. Reduce set of active indices
               neg_grad   = neg_grad[np.bitwise_not(out_test2)]
               ind_active = ind_active[np.bitwise_not(out_test2)]
               nb_screen  = np.sum(out_test2)
               n_active  -= nb_screen

               gamma_returned = gamma_returned[nb_screen:]

               # schedule next test
               gap_last_test = float(gap)
               lip = update_Lip()

         elif is_test_1:
            out_test1 = screening_test_1.apply_test(np.abs(neg_grad), gap, lbd, vecgamma[:n_active], coeff_dual_scaling=coeff_dual_scaling, index=index_sort_neg_grad)
            
            if np.any(out_test1):
               # 2a. Set entry to 0
               vecx_hat[ind_active[out_test1]] = 0.

               # 2b. Reduce set of active indices
               neg_grad       = neg_grad[np.bitwise_not(out_test1)]
               ind_active     = ind_active[np.bitwise_not(out_test1)]
               nb_screen      = np.sum(out_test1)
               n_active      -= nb_screen 

               gamma_returned = gamma_returned[nb_screen:]
               lip = update_Lip()


      # ------------------
      #      Alg steps
      # ------------------

      if algParameters.accelerated:
         # a. Gradient step
         x_copy = np.copy(vecx_hat[ind_active])

         vecx_hat[ind_active] = x_tilde[ind_active] + \
            (Aty[ind_active] - matA[:, ind_active].T @ (matA[:, ind_active] @ x_tilde[ind_active])) / lip

         # b. Proximal step
         vecx_hat[ind_active] = prox_owl(vecx_hat[ind_active], (lbd / lip) * vecgamma[:n_active])

         # c. fista stuff
         beta_buf = float(beta_k)
         beta_k = (1. + np.sqrt(1. + 4. * beta_buf**2)) / 2.
         x_tilde[ind_active] = vecx_hat[ind_active] + ((beta_buf - 1.) / beta_k) * (vecx_hat[ind_active] - x_copy)

      else:
         # a. Gradient step
         vecx_hat[ind_active] += neg_grad / lip

         # b. Prox step
         vecx_hat[ind_active] = prox_owl(vecx_hat[ind_active], (lbd / lip) * vecgamma[:n_active])

      # Iteration step
      it += 1
      
      if algParameters.save_nactive:
         vec_n_active.append(n_active)

   time_run = ellapsed_time()


   return {
      "sol": best_vecx,
      "cost_function": np.array(vec_cost_func),
      "dual_function": np.array(vec_dual_func),
      "gap": gap,
      "time_run": time_run,
      "nb_it": it,
      "vec_n_active": np.array(vec_n_active),
   }


def __assert(vecy, matA, lbd, vecgamma, algParameters):
   """
   """

   assert(isinstance(vecy, np.ndarray))
   assert(len(vecy.shape) == 1)

   assert(isinstance(matA, np.ndarray))
   assert(len(matA.shape) == 2)
   assert(matA.shape[0] == vecy.shape[0])

   assert(isinstance(lbd, float))
   assert(lbd > 0)

   assert(isinstance(vecgamma, np.ndarray))
   assert(len(vecgamma.shape) == 1)
   assert(vecgamma.shape[0] == matA.shape[1])

   assert(isinstance(algParameters, SlopeParameters))





if __name__ == '__main__':

   # -------------------------
   #     Data generation
   # -------------------------
   m = 10
   n = 15

   vecy = np.random.rand(m)
   matA = np.random.rand(m, n)
   for j in range(n):
      matA[:, j] /= np.linalg.norm(matA[:, j], 2)


   # -------------------------
   #        Learning
   # -------------------------

   lbd = 1.
   vecgamma = np.linspace(1., .1, n)
   params = SlopeParameters()

   slope_gp(vecy, matA, lbd, vecgamma, params)
