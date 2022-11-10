# -*- coding: utf-8 -*-
from dataclasses import dataclass
import enum

import numpy as np

from slopescreening.screening.ascreening import AbstractGapScreening


class EnumLipchitzOptions(enum.Enum):
   EXACT        = enum.auto()
   GERSHGORIN   = enum.auto()
   BACKTRACKING = enum.auto()


class DualScalingOptions(enum.Enum):
   EXACT     = enum.auto()
   BAO_ET_AL = enum.auto()


class MyList:
   def __getitem__(self, key):
      return float(key)


@dataclass
class SlopeParameters(object):
   max_it: int = 1e3
   gap_stopping: float = 1e-9
   time_stopping: float = 1e2
   accelerated: bool = False
   vecx_init: np.ndarray = None
   verbose: bool = True

   dual_scaling: DualScalingOptions = DualScalingOptions.EXACT

   eval_gap: bool   = True
   eval_gap_it: int = 1

   lipchitz_constant: float = None
   lipchitz_update: EnumLipchitzOptions = EnumLipchitzOptions.EXACT

   coherence: float = 1.
   coherence_function: np.ndarray = MyList()

   screening1: AbstractGapScreening = None
   screening2: AbstractGapScreening = None

   save_nactive: bool = False
   save_pfunc:  bool = False
   save_dfunc:  bool = False



# @dataclass
# class SlopeParameters(AlgParameters):




