# -*- coding: utf-8 -*-
import numpy as np
import scipy.fftpack


def generate_dic(str_dic, m, n, bnormalized=True):
   """Generate dictionaries 
    
   Parameters
   ----------
   str_dic : str
      str representing type of dictionaries
      can be ['gaussian', 'uniform', 'dct', 'toeplitz']
   m : int
      number of rows
   n : int
      number of columns
   bnormalized : bool
      true if columns are unit-norm

   Returns
   -------
   matA : np.ndarray
      dictionary matrix in 'Fortran' order (column major)
      size [m, n]
   """

   if 'gaussian' in str_dic:
      sp = str_dic.split(' ')
      if len(sp) == 1:
         rho = 0
      else:
         rho = float(sp[1])

      matA = _gaussian(m, n, rho)

   elif str_dic == "uniform":
      matA = _uniform(m, n)

   elif str_dic == "dct":
      matA = _dct(m, n)
   
   elif str_dic == "toeplitz":
      matA = _toeplitz(m, n)

   elif str_dic =="pulse":
      #Defined in Trop'
      matA = _pulse(m, n)

   else:
      raise "not recognized"

   if bnormalized:
      for j in range(n):
         matA[:, j] /= np.linalg.norm(matA[:, j], 2)

   return matA


def _gaussian(m, n, rho):
   """Generate dictionaries whose column are 
   multivariate Gaussian with covariance matrix
   \\Sigma satisfying $\\Sigma_{i,j} = 1$ if $i=j$ and $rho$ otherwise
   
   Parameters
   ----------
   m : int
      number of rows
   n : int
      number of columns
   corr : positive float
      non-diagonal entries of the correlation matrix

   Returns
   -------
   matA : np.ndarray
      matrix dictionary in 'Fortran order' (column major)
   """

   matCov = np.zeros((m, m))
   for i in range(m):
      for j in range(m):
         matCov[i, j] = .7**(np.abs(i - j))
 
   matA = np.random.multivariate_normal(np.zeros(m), matCov, n).T

   return matA


def _uniform(m, n):
   # -- generate data
   matA = np.random.rand(n, m).T

   return matA


def _dct(m, n):
   indices = np.random.permutation(n)
   indices = indices[:m]

      # Coding matrix known to have good properties
   matA = np.copy(scipy.fftpack.dct(np.eye(n)))
   matA = matA[indices, :]

   return np.copy(matA, order='F')


def _toeplitz(m, n):
   gauss = lambda t: np.exp(-.5 * (t**2))

   ranget = np.linspace(-10, 10, m)
   offset = 3.
   rangemu = np.linspace(np.min(ranget)+offset, np.max(ranget)-offset, n)

   matA = np.zeros((n, m)).T
   for j in range(n):
      matA[:, j] = gauss(ranget - rangemu[j])

   return matA


def _pulse(m, n):
   matA = np.zeros((n, m)).T


if __name__ == '__main__':
   
   # Check if all dictionaries are in Fortran order
   m = 10
   n = 15

   list_dic = ["gaussian", "uniform", "dct", "toeplitz"]

   for str_dic in list_dic:
      print(f"-- {str_dic} --")
      matA = generate_dic(str_dic, m, n, True)

      print(f"is fotran: {np.isfortran(matA)}")
      print("\n")
