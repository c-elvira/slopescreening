# -*- coding: utf-8 -*-
from dataclasses import dataclass
import enum

import numpy as np

from src.screening.ascreening import AbstractGapScreening


class EnumLipchitzOptions(enum.Enum):
	EXACT 		 = enum.auto()
	GERSHGORIN   = enum.auto()


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

    lipchitz_constant: float = None
    lipchitz_update: EnumLipchitzOptions = EnumLipchitzOptions.EXACT

    coherence: float = 1.
    coherence_function: np.ndarray = MyList()

    screening1: AbstractGapScreening = None
    screening2: AbstractGapScreening = None
    screening_it_div: float = 2.

    save_nactive = False
