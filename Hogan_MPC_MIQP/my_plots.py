#!/usr/bin/env python3.8
## Author: Joao Moura
## Contact: jpousad@ed.ac.uk
## Date: 16/12/2020
## -------------------------------------------------------------------
## Description:
## 
## Functions for plotting trajectories
## -------------------------------------------------------------------

## Import Libraries
#  -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import casadi as cs

## Return patches
#  -------------------------------------------------------------------
def get_patches_for_square_slider_and_cicle_pusher(ax, pfunc, Rfunc, x_data, sq_sd, r_pusher):
    x0 = x_data[:,0]
    d0 = np.array(cs.mtimes(Rfunc(x0),[-sq_sd/2, -sq_sd/2, 0]).T)[0]
    slider = patches.Rectangle(x0[0:2]+d0[0:2], sq_sd, sq_sd, x0[2])
    pusher = patches.Circle(np.array(pfunc(x0)), radius=r_pusher, color='black')
    ax.add_patch(slider)
    ax.add_patch(pusher)
    return slider, pusher
#  -------------------------------------------------------------------

## Return patches
#  -------------------------------------------------------------------
def animate_square_slider_and_circle_pusher(i, slider, pusher, ax, pfunc, Rfunc, x_data, sq_sd):
    xi = x_data[:,i]
    # distance between centre of square reference corner
    di = np.array(cs.mtimes(Rfunc(xi),[-sq_sd/2, -sq_sd/2, 0]).T)[0]
    # square reference corner
    ci = xi[0:3] + di
    # compute transformation with respect to rotation angle xi[2]
    trans_ax = ax.transData
    coords = trans_ax.transform(ci[0:2])
    trans_i = transforms.Affine2D().rotate_around(coords[0], coords[1], xi[2])
    # Set changes
    slider.set_transform(trans_ax+trans_i)
    slider.set_xy([ci[0], ci[1]])
    pusher.set_center(np.array(pfunc(xi)))
    return []
#  -------------------------------------------------------------------

## Plot nominal trajectory
#  -------------------------------------------------------------------
def plot_nominal_traj(x_data, y_data,
        margin_width=0.1,
        window_title='Nominal Trajectory',
        plot_title='Pusher-Slider Motion Animation'):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(window_title)
    ax.plot(x_data, y_data, color='red', linewidth=2.0, linestyle='dashed')
    # plot initial and final positions
    ax.plot(x_data[0], y_data[0], x_data[-1], y_data[-1], marker='o', color='red')
    # ax limits
    ax.set_xlim((np.min(x_data)-margin_width,np.max(x_data)+margin_width))
    ax.set_ylim((np.min(y_data)-margin_width,np.max(y_data)+margin_width))
    # ax settings
    ax.set_autoscale_on(False)
    ax.grid();
    ax.set_aspect('equal', 'box')
    ax.set_title(plot_title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    return fig, ax
