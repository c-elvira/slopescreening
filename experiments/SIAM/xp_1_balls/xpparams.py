import numpy as np

# Algorithmic parameters
stopping_gap = 1e-14

# -- Screening parameter --
# vec_offsets = np.linspace(0, setup.max_offset, setup.nb_point)
nb_point   = 2000
max_offset = 2
vec_offsets = max_offset * np.logspace(-7., .0, num=nb_point)