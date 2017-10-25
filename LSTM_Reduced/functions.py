import numpy as np
import sklearn.preprocessing as sk
from sklearn.metrics import mean_squared_error
import numpy.matlib as matlib
import sys
from sklearn.neighbors import KernelDensity
import scipy as sp
import pickle



def lorenz(t0, u0, sigma, rho, beta):
    dudt = np.zeros(np.shape(u0))
    dudt[0] = sigma * (u0[1]-u0[0])
    dudt[1] = u0[0] * (rho-u0[2]) - u0[1]
    dudt[2] = u0[0] * u0[1] - beta*u0[2]
    return dudt













