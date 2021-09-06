# -*- coding: utf-8 -*-
import numpy as np

from src.screening.ascreening import AbstractGapScreening


class GapSphereSingleTest(AbstractGapScreening):
   """ Single test safe sphere screening test for Slope
   """
   def __init__(self):
      super(GapSphereSingleTest, self).__init__()

      self.name = "single test"

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

      radius = np.sqrt(2 * gap)
      return Atu_abs < lbd * vec_gammas[-1] - radius - offset_radius
