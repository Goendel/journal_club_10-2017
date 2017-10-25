#!/usr/bin/env python
from __future__ import print_function
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import os
import sys
import argparse
from config import Config as conf

mpl.rcParams['legend.fontsize'] = 10

with open(conf.prediction_results_path, "rb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    data = pickle.load(file)
    predicted_evolution_en_mean = data["predicted_evolution_en_mean"]
    true_evolution = data["true_evolution"]
    conf = data["conf"]
    del data

N_plot_max = np.shape(true_evolution)[0]
n_ics = np.min([np.shape(true_evolution)[2], 20])
for N_plot in [300, 600, 1000, N_plot_max]:
    for ic in range(n_ics):
        u = predicted_evolution_en_mean[:,:,ic]
        u_true = true_evolution[:,:,ic]
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(np.arange(0,N_plot)*conf.dt, u_true[:N_plot,0], "g--", label='Lorenz Trajectory')
        ax.plot(np.arange(0,N_plot)*conf.dt, u[:N_plot,0], "r--", label='Predicted')
        ax.legend()
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$X_1$')
        plt.savefig("./Prediction_Figures/Trajectories_plot_N{:d}_IC{:d}.png".format(N_plot, ic), bbox_inches="tight")
        plt.close()


N_plot_max = np.shape(true_evolution)[0]
n_ics = np.min([np.shape(true_evolution)[2], 5])
for N_plot in [1000]:
    for ic in range(n_ics):
        u = predicted_evolution_en_mean[:,:,ic]
        u_true = true_evolution[:,:,ic]
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(u_true[:N_plot-1,0], u_true[1:N_plot,0], "g--", label='Lorenz')
        ax.plot(u[:N_plot-1,0], u[1:N_plot,0], "r--", label='Predicted')
        ax.legend()
        ax.set_xlabel(r'$X_1^{t}$')
        ax.set_ylabel(r'$X_1^{t+1}$')
        plt.savefig("./Prediction_Figures/Attractor.png".format(N_plot, ic), bbox_inches="tight")
        plt.close()




