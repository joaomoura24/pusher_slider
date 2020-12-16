## Author: Joao Moura
## Contact: jpousad@ed.ac.uk
## Date: 15/12/2020
## -------------------------------------------------------------------
## Description:
## 
## Functions for outputting different nominal trajectories
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## Import libraries
## -------------------------------------------------------------------
import numpy as np
import sys
# import casadi as cs

## Generate Nominal Trajectory (line)
def generate_traj_line(x_f, y_f, N):
    x_nom = np.linspace(0.0, x_f, N)
    y_nom = np.linspace(0.0, y_f, N)
    return x_nom, y_nom
def generate_traj_circle(theta_i, theta_f, radious, N):
    s = np.linspace(theta_i, theta_f, N)
    x_nom = radious*np.cos(s)
    y_nom = radious*np.sin(s)
    # initial position at the origin
    x_nom -= x_nom[0]
    y_nom -= y_nom[0]
    return x_nom, y_nom
def generate_traj_eight(side_lenght, N):
    s = np.linspace(0.0, 2*np.pi, N)
    x_nom = side_lenght*np.sin(s)
    y_nom = side_lenght*np.sin(s)*np.cos(s)
    return x_nom, y_nom
def compute_nomState_from_nomTraj(x_data, y_data, dt):
    # assign two first state trajectories
    x0_nom = x_data
    x1_nom = y_data
    # compute diff for plannar traj
    Dx0_nom = np.diff(x0_nom)
    Dx1_nom = np.diff(x1_nom)
    # compute traj angle 
    x2_nom = np.arctan2(Dx1_nom, Dx0_nom);
    x2_nom = np.append(x2_nom, x2_nom[-1])
    Dx2_nom = np.diff(x2_nom)
    # specify angle of the pusher relative to slider
    x3_nom = np.zeros(x0_nom.shape)
    Dx3_nom = np.diff(x3_nom)
    # stack state and derivative of state
    x_nom = np.vstack((x0_nom, x1_nom, x2_nom, x3_nom))
    dx_nom = np.vstack((Dx0_nom, Dx1_nom, Dx2_nom, Dx3_nom))/dt
    return x_nom, dx_nom
