# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import numpy as np


class AbstractGapScreening(ABC):
   """docstring for Screening"""
   def __init__(self):
      super(AbstractGapScreening, self).__init__()

      self.name = "NONAME"

   def get_name(self):
      return self.name
   

   @abstractmethod
   def apply_test(self, Atu_abs, gap, lbd, vec_gammas, coeff_dual_scaling=1., offset_radius=0, index=None) -> np.ndarray:
      """ GAP Sphere screening test detailed in Cédric's node
      
      Parameters
      ----------
      Atu_abs : np.ndarray
         vector |matA.T @ vecu|
         vecu is not necessarily dual admissible
         In that case, the dual scaling factor must be provided
         size [n]
      gap : positive float
         duality gap
      lbd : positive float
         regularization parameters
      vec_gammas : np.ndarray
         slope parameters
         size [n,]
      coeff_dual_scaling : positif float
         If coeff_dual_scaling is not feasible, dual scaling factor
         such taht vecu / coeff_dual_scaling os dual feasible
         Here for code optimization purposes
         Default value is 1. (vecu is feasible)
      offset_radius : float
         additive term added to the redius
      index : np.ndarray
         Array of indices that sort Atu in absolute value
         (unused here)
         default is None

      Returns
      -------
      calI_screen : np.ndarray
         vector of boolean
         True if screening test passes and False otherwise
         size [n,]
      """

      pass