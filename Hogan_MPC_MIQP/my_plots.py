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

## Plot function
#  -------------------------------------------------------------------
def plot_traj_static(x_data, y_data, margin_width=0.1):
    fig_traj = plt.figure()
    fig_traj.canvas.set_window_title('Nominal trajectory')
    ax = fig_traj.add_subplot(111, aspect='equal', autoscale_on=False, \
            xlim=(np.min(x_data)-margin_width,np.max(x_data)+margin_width), \
            ylim=(np.min(y_data)-margin_width,np.max(y_data)+margin_width) \
    )
    # plot trajectory with dashed line
    ax.plot(x_data, y_data, color='red', linewidth=2.0, linestyle='dashed')
    # plot initial and final positions
    ax.plot(x_data[0], y_data[0], x_data[-1], y_data[-1], marker='o', color='red')
    ax.grid();
    ax.set_aspect('equal', 'box')
    ax.set(xlabel='x [m]', ylabel='y [m]', title='Nominal trajectory')
    plt.show(block=False)
