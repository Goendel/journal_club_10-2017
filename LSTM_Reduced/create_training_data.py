#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as sk
import os # for saving
import pickle
import os
import sys
import argparse
import random as rand
import math
from scipy.integrate import ode


with open("../Data_Generation/Data/lorenz_data.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u = data["u"]
    sigma = data["sigma"]
    beta = data["beta"]
    rho = data["rho"]
    dt = data["dt"]

with open("../Data_Generation/Data/lorenz_data_ic.pickle", "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    u_IC = data["u_IC"]

dudt = (u[1:]-u[:-1])/dt
u = u[:-1,:]

dudt = dudt[:1000,:]
u = u[:1000,:]
u_IC = u_IC[:1000,:]

loss_weights = np.float32(np.reshape(np.array([1]), (1,-1)))
input_sequence = np.reshape(u[:,0], (-1,1))
target_sequence = np.reshape(dudt[:,0], (-1,1))
input_initial_conditions = u_IC

data = {"input_initial_conditions":input_initial_conditions,
        "input_sequence":input_sequence,
        "target_sequence":target_sequence,
        "loss_weights":loss_weights,
        "dt":dt,"sigma":sigma,
        "beta":beta,
        "rho":rho,}

with open("./Training_Data/lorenz_training_data_reduced.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)



