# -*- coding: utf-8 -*-
import numpy as np
from sklearn.isotonic import isotonic_regression


def prox_owl(vec_x, vec_slope_weights):
    """ Proximity operator for the Slope regularizer proposed by Zeng & Figueiredo
    
    Implementation of the proximal operator proposed by Zang and Figueiredo in [1, sec III.A]
    relying on conic projection.
    
    [1] X. Zeng, M. Figueiredo, The ordered weighted L1 norm: Atomic formulation, dual norm,
    and projections. https://arxiv.org/abs/1409.4271

    Parameters
    ----------
    vec_x : np.ndarray
        input vector
        size [n,]
    vec_slope_weights : np.ndarray
        vector of Slope parameters
        size [n,]

    Returns
    -------
    prox : np.ndarray
        proximity operator
    """

    # wlog operate on absolute values
    vec_x_abs = np.abs(vec_x)
    ix = np.argsort(vec_x_abs)[::-1]
    vec_x_abs = vec_x_abs[ix]

    # project to K+ (monotone non-negative decreasing cone)
    # La projection est en O(n) -- je ne sais pas en dire plus
    vec_x_abs = isotonic_regression(vec_x_abs - vec_slope_weights, y_min=0, increasing=False)

    # undo the sorting
    inv_ix = np.zeros_like(ix)
    inv_ix[ix] = np.arange(len(vec_x))
    vec_x_abs = vec_x_abs[inv_ix]

    return np.sign(vec_x) * vec_x_abs
