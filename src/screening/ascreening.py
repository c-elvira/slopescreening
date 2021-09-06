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
   def apply_test(self, Atu_abs, gap, lbd, vec_gammas, offset_radius=0, index=None) -> np.ndarray:
      """ GAP Sphere screening test detailed in Cédric's node
      
      Parameters
      ----------
      Atu_abs : np.ndarray
         vector |matA.T @ vecu| where vecu is dual admissible
         size [n]
      gap : positive float
         duality gap
      lbd : positive float
         regularization parameters
      vec_gammas : np.ndarray
         slope parameters
         size [n,]
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