# Author: Joao Moura
# Date: 21/08/2020
#  -------------------------------------------------------------------
# Description:
#  This script implements a non-linear program (NLP) model predictive controller (MPC)
#  for tracking a trajectory of a square slider object with a single
#  and sliding contact pusher.
#  -------------------------------------------------------------------

#  import libraries
#  -------------------------------------------------------------------
import sys
import yaml
import numpy as np
import pandas as pd
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
# cc or mi
# contact = 'cc'
contact = 'mi'
num_rep = 10
ang_dist = 3.0
# ang_dist = 1.5
# ang_dist = 0.
#  -------------------------------------------------------------------

# Get config files
#  -------------------------------------------------------------------
config_file_name = '../config/tracking_noise_' + contact + '_config.yaml'
with open(config_file_name, 'r') as configFile:
    tracking_config = yaml.load(configFile, Loader=yaml.FullLoader)
with open('../config/nom_config.yaml', 'r') as configFile:
    planning_config = yaml.load(configFile, Loader=yaml.FullLoader)
#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
T = 10  # time of the simulation is seconds
freq = 25  # number of increments per second
# N_MPC = 25  # time horizon for the MPC controller
# x_init_val = [-0.01, 0.03, 30*(np.pi/180.), 0]
x_init_val = [0., 0., 0., 0.]
show_anim = False
show_plot = False
save_to_file = True
#  -------------------------------------------------------------------
# Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq  # sampling time
N = int(T*freq)  # total number of iterations
Nidx = int(N)
#  -------------------------------------------------------------------

# define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
        tracking_config['dynamics'],
        tracking_config['TO']['contactMode']
)
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
# creation of files
column_names_state = ['x_nom', 'y_nom', 'theta_nom', 'psi_nom',
                'x_opt', 'y_opt', 'theta_opt', 'psi_opt']
column_names_time = ['comp_time']
#  -------------------------------------------------------------------

for N_MPC in range(5, 76, 1):

    # Generate Nominal Trajectory
    #  -------------------------------------------------------------------
    X_goal = tracking_config['TO']['X_goal']
    x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.2, N, N_MPC)
    #  -------------------------------------------------------------------
    # stack state and derivative of state
    X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
    #  ------------------------------------------------------------------

    # Compute nominal actions for sticking contact
    #  ------------------------------------------------------------------
    dynNom = sliding_pack.dyn.Sys_sq_slider_quasi_static_ellip_lim_surf(
            planning_config['dynamics'],
            planning_config['TO']['contactMode']
    )
    optObjNom = sliding_pack.to.buildOptObj(
            dynNom, N+N_MPC, planning_config['TO'], X_nom_val, dt=dt)
    resultFlag, X_nom_val_opt, U_nom_val_opt, _, _, _ = optObjNom.solveProblem(
            0, [0., 0., 0., 0.])
    if dyn.Nu > dynNom.Nu:
        U_nom_val_opt = cs.vertcat(
                U_nom_val_opt,
                cs.DM.zeros(np.abs(dyn.Nu - dynNom.Nu), N+N_MPC-1))
    elif dynNom.Nu > dyn.Nu:
        U_nom_val_opt = U_nom_val_opt[:dyn.Nu, :]
    #  ------------------------------------------------------------------

    # define optimization problem
    #  -------------------------------------------------------------------
    optObj = sliding_pack.to.buildOptObj(
            dyn, N_MPC, tracking_config['TO'],
            X_nom_val, U_nom_val_opt, dt=dt,
    )
    #  -------------------------------------------------------------------

    df_s = pd.DataFrame(columns=column_names_state)
    df_t = pd.DataFrame(columns=column_names_time)
    for i_runs in range(num_rep):
        # Initialize variables for plotting
        #  -------------------------------------------------------------------
        X_plot = np.empty([dyn.Nx, Nidx])
        U_plot = np.empty([dyn.Nu, Nidx-1])
        del_plot = np.empty([dyn.Nz, Nidx-1])
        X_plot[:, 0] = x_init_val
        X_future = np.empty([dyn.Nx, N_MPC, Nidx])
        comp_time = np.empty((Nidx-1, 1))
        success = np.empty(Nidx-1)
        cost_plot = np.empty((Nidx-1, 1))
        #  -------------------------------------------------------------------

        # Set arguments and solve
        #  -------------------------------------------------------------------
        x0 = x_init_val
        for idx in range(Nidx-1):
            # print('-------------------------')
            # print(idx)
            # ---- solve problem ----
            resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObj.solveProblem(idx, x0)
            # print(f_opt)
            # ---- update initial state (simulation) ----
            u0 = u_opt[:, 0].elements()
            # x0 = x_opt[:,1].elements()
            epsilon_dist = [0., 0., np.random.uniform(-ang_dist, ang_dist), 0.]
            x0 = (x0 + (dyn.f(x0, u0) + epsilon_dist)*dt).elements()
            # ---- store values for plotting ----
            comp_time[idx] = t_opt
            success[idx] = resultFlag
            cost_plot[idx] = f_opt
            X_plot[:, idx+1] = x0
            U_plot[:, idx] = u0
            X_future[:, :, idx] = np.array(x_opt)
            if dyn.Nz > 0:
                del_plot[:, idx] = del_opt[:, 0].elements()
        #  -------------------------------------------------------------------
        # show sparsity pattern
        # sliding_pack.plots.plot_sparsity(cs.vertcat(*opt.g), cs.vertcat(*opt.x), xu_opt)
        #  -------------------------------------------------------------------

        # Animation
        #  -------------------------------------------------------------------
        plt.rcParams['figure.dpi'] = 150
        if show_anim:
            #  ---------------------------------------------------------------
            fig, ax = sliding_pack.plots.plot_nominal_traj(
                        x0_nom[:Nidx], x1_nom[:Nidx])
            # add computed nominal trajectory
            X_nom_val_opt = np.array(X_nom_val_opt)
            ax.plot(X_nom_val_opt[0, :], X_nom_val_opt[1, :], color='blue',
                    linewidth=2.0, linestyle='dashed')
            # plot final trajectory
            X_plot = np.array(X_plot)
            ax.plot(X_plot[0, :], X_plot[1, :], color='pink',
                    linewidth=1.0)
            # set window size
            fig.set_size_inches(8, 6, forward=True)
            plt.title('ang_dist = ' + repr(ang_dist) + '; run = ' + repr(i_runs))
            # get slider and pusher patches
            dyn.set_patches(ax, X_plot)
            # call the animation
            # ani = animation.FuncAnimation(
            #         fig,
            #         dyn.animate,
            #         fargs=(ax, X_plot, X_future),
            #         frames=Nidx-1,
            #         interval=dt*1000,  # microseconds
            #         blit=True,
            #         repeat=False,
            # )
            # # to save animation, uncomment the line below:
            # # ani.save('MPC_NLP_eight.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
            #  -------------------------------------------------------------------
            plt.show()
            #  -------------------------------------------------------------------
        #  -------------------------------------------------------------------

        if save_to_file:
            #  Save data to file using pandas
            #  -------------------------------------------------------------------
            df_time = pd.DataFrame(
                            np.array(comp_time),
                            columns=column_names_time)
            df_state = pd.DataFrame(
                            np.concatenate((
                                np.array(X_nom_val[:, :Nidx]).transpose(),
                                np.array(X_plot).transpose()
                                ), axis=1),
                            columns=column_names_state)
            df_s = df_s.append(df_state, ignore_index=True)
            df_t = df_t.append(df_time, ignore_index=True)
            #  -------------------------------------------------------------------

    if save_to_file:
        #  Save data to file using pandas
        #  -------------------------------------------------------------------
        df_s.to_csv('data_th/{}/tracking_time_horizon_{}_{}_state.csv'.format(
            contact, N_MPC, contact), float_format='%.5f')
        df_t.to_csv('data_th/{}/tracking_time_horizon_{}_{}_time.csv'.format(
            contact, N_MPC, contact), float_format='%.5f')
        #  -------------------------------------------------------------------

if show_plot:
    # Plot Optimization Results
    #  -------------------------------------------------------------------
    fig, axs = plt.subplots(3, 4, sharex=True)
    fig.set_size_inches(10, 10, forward=True)
    t_Nx = np.linspace(0, T, N)
    t_Nu = np.linspace(0, T, N-1)
    t_idx_x = t_Nx[0:Nidx]
    t_idx_u = t_Nx[0:Nidx-1]
    ctrl_g_idx = dyn.g_u.map(Nidx-1)
    ctrl_g_val = ctrl_g_idx(U_plot, del_plot)
    #  -------------------------------------------------------------------
    # plot position
    for i in range(dyn.Nx):
        axs[0, i].plot(t_Nx, X_nom_val[i, 0:N].T, color='red',
                       linestyle='--', label='nom')
        axs[0, i].plot(t_Nx, X_nom_val_opt[i, 0:N].T, color='blue',
                       linestyle='--', label='plan')
        axs[0, i].plot(t_idx_x, X_plot[i, :], color='orange', label='mpc')
        handles, labels = axs[0, i].get_legend_handles_labels()
        axs[0, i].legend(handles, labels)
        axs[0, i].set_xlabel('time [s]')
        axs[0, i].set_ylabel('x%d' % i)
        axs[0, i].grid()
    #  -------------------------------------------------------------------
    # plot computation time
    axs[1, 0].plot(t_idx_u, comp_time, color='b')
    handles, labels = axs[1, 0].get_legend_handles_labels()
    axs[1, 0].legend(handles, labels)
    axs[1, 0].set_xlabel('time [s]')
    axs[1, 0].set_ylabel('comp time [s]')
    axs[1, 0].grid()
    #  -------------------------------------------------------------------
    # plot computation cost
    axs[1, 1].plot(t_idx_u, cost_plot, color='b', label='cost')
    handles, labels = axs[1, 1].get_legend_handles_labels()
    axs[1, 1].legend(handles, labels)
    axs[1, 1].set_xlabel('time [s]')
    axs[1, 1].set_ylabel('cost')
    axs[1, 1].grid()
    #  -------------------------------------------------------------------
    # plot extra variables
    for i in range(dyn.Nz):
        axs[1, 2].plot(t_idx_u, del_plot[i, :].T, label='s%d' % i)
    handles, labels = axs[1, 2].get_legend_handles_labels()
    axs[1, 2].legend(handles, labels)
    axs[1, 2].set_xlabel('time [s]')
    axs[1, 2].set_ylabel('extra vars')
    axs[1, 2].grid()
    #  -------------------------------------------------------------------
    # plot constraints
    for i in range(dyn.Ng_u):
        axs[1, 3].plot(t_idx_u, ctrl_g_val[i, :].T, label='g%d' % i)
    handles, labels = axs[1, 3].get_legend_handles_labels()
    axs[1, 3].legend(handles, labels)
    axs[1, 3].set_xlabel('time [s]')
    axs[1, 3].set_ylabel('constraints')
    axs[1, 3].grid()
    #  -------------------------------------------------------------------
    # plot actions
    for i in range(dyn.Nu):
        axs[2, i].plot(t_Nu, U_nom_val_opt[i, 0:N-1].T, color='blue',
                       linestyle='--', label='plan')
        axs[2, i].plot(t_idx_u, U_plot[i, :], color='orange', label='mpc')
        handles, labels = axs[2, i].get_legend_handles_labels()
        axs[2, i].legend(handles, labels)
        axs[2, i].set_xlabel('time [s]')
        axs[2, i].set_ylabel('u%d' % i)
        axs[2, i].grid()
    #  -------------------------------------------------------------------
