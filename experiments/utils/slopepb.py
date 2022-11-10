import numpy as np

from slopescreening.slope import primal_func, dual_func
from slopescreening.utils import get_lambda_max


class SlopePb(object):
   """docstring for SlopePb"""
   def __init__(self, matA, vecy, vec_gammas, ratio_lbd, lbdmax=None):
      super(SlopePb, self).__init__()
      self.matA = matA
      self.vecy = vecy
      self.vec_gammas = vec_gammas
      self.ratio_lbd = ratio_lbd
      
      if lbdmax is None:
         self.lbdmax = get_lambda_max(vecy, matA, vec_gammas)
      else:
         self.lbdmax = lbdmax

      self.lbd = ratio_lbd * self.lbdmax


   def make_dual_scaling(self, vecr):

      beta_dual = np.sort(np.abs(self.matA.T @ vecr))[::-1]
      beta_dual = np.cumsum(beta_dual) / \
         np.cumsum(self.lbd * self.vec_gammas)

      return vecr / np.max(beta_dual)


   def make_screening_quanties(self, vecx):
      """
      """

      # residual error
      vecu = self.vecy - self.matA @ vecx

         # dual scaling
      vecu = self.make_dual_scaling(vecu)

      pval = primal_func(self.vecy, self.matA @ vecx, vecx, 
         self.lbd, self.vec_gammas
      )
      dval = dual_func(self.vecy, 
         np.linalg.norm(self.vecy, 2)**2, vecu
      )

      gap = np.abs(pval - dval)

      Atu = self.matA.T @ vecu

      return vecu, Atu, gap


   def eval_gap(self, vecx):
      """
      """

      # residual error
      vecu = self.vecy - self.matA @ vecx

         # dual scaling
      vecu = self.make_dual_scaling(vecu)

      pval = primal_func(self.vecy, self.matA @ vecx, vecx, 
         self.lbd, self.vec_gammas
      )
      dval = dual_func(self.vecy, 
         np.linalg.norm(self.vecy, 2)**2, vecu
      )

      return np.abs(pval - dval)