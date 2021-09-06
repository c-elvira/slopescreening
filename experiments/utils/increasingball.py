import time

import numpy as np

from src.utils import get_lambda_max


def test_increasing_ball(list_tests, vec_offsets, slopepb, vecx):

   nb_test = len(list_tests)
   mat_out = np.zeros((nb_test, vec_offsets.size))

   vecu, Atu, gap = slopepb.make_screening_quanties(
      vecx
   )

   list_time = [0 for i in range(nb_test)]

   # Boucle
   for i_offset, offset in enumerate(vec_offsets):
      for i_test, test in enumerate(list_tests):
         # Test
         t_start = time.time()
         out = test.apply_test(
            np.abs(Atu), 
            gap, 
            slopepb.lbd, 
            slopepb.vec_gammas, 
            offset_radius=offset
         )
         list_time[i_test] += time.time() - t_start
         
         # save
         mat_out[i_test, i_offset] = np.sum(out)

   # out
   return mat_out, list_time