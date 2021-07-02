## Author: Joao Moura
## Date: 21/08/2020
#  -------------------------------------------------------------------
## Description:
#  This script implements a non-linear program (NLP) model predictive controller (MPC)
#  for tracking a trajectory of a square slider object with a single
#  and sliding contact pusher.
#  -------------------------------------------------------------------

## Import Libraries
#  -------------------------------------------------------------------
import sys
import numpy as np
# import casadi as cs
# import casadi
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import matplotlib.animation as animation
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

# Set Problem constants
#  -------------------------------------------------------------------
a = 0.09 # side dimension of the square slider in meters
T = 12 # time of the simulation is seconds
freq = 25 # number of increments per second
r_pusher = 0.01 # radius of the cylindrical pusher in meter
miu_p = 0.4  # friction between pusher and slider
N_MPC = 12 # time horizon for the MPC controller
x_init_val = [-0.01, 0.03, 30*(np.pi/180.), 0]
f_lim = 0.3 # limit on the actuations
psi_dot_lim = 3.0 # limit on the actuations
psi_lim = 40*(np.pi/180.0)
# solver_name = 'ipopt'
solver_name = 'snopt'
# solver_name = 'gurobi'
# solver_name = 'qpoases'
no_printing = True
code_gen = False
show_anim = True
#  -------------------------------------------------------------------
# Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq # sampling time
N = int(T*freq) # total number of iterations
Nidx = int(N)
# Nidx = 3
#  -------------------------------------------------------------------

# define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.System_square_slider_quasi_static_ellipsoidal_limit_surface(
        mode='sticking_contact',
        # mode='sliding_contact',
        slider_dim=a,
        pusher_radious=r_pusher,
        miu=miu_p,
        f_lim=f_lim,
        psi_dot_lim=psi_dot_lim,
        psi_lim=psi_lim
)
#  -------------------------------------------------------------------

## Generate Nominal Trajectory
#  -------------------------------------------------------------------
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.5, 0.0, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.5, 0.3, N, N_MPC)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.1, N, N_MPC)
x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.2, N, N_MPC)
#  -------------------------------------------------------------------
# stack state and derivative of state
X_nom_val, _ = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  ------------------------------------------------------------------

# define optimization problem
#  -------------------------------------------------------------------
optObj = sliding_pack.nlp.MPC_nlpClass(
        dyn, N_MPC, X_nom_val, dt=dt)
#  -------------------------------------------------------------------
optObj.buildProblem(solver_name, code_gen, no_printing)
#  -------------------------------------------------------------------

## Initialize variables for plotting
#  -------------------------------------------------------------------
X_plot = np.empty([dyn.Nx, Nidx])
U_plot = np.empty([dyn.Nu, Nidx-1])
del_plot = np.empty([dyn.Nz, Nidx-1])
X_plot[:,0] = x_init_val
X_future = np.empty([dyn.Nx, N_MPC, Nidx])
comp_time = np.empty(Nidx-1)
success = np.empty(Nidx-1)
cost_plot = np.empty(Nidx-1)
#  -------------------------------------------------------------------

## Set arguments and solve
#  -------------------------------------------------------------------
x0 = x_init_val
# x0 = [0.0, 0.0, 0.0, 0.0]
u0 = [0.0, 0.0, 0.0, 0.0]
for idx in range(Nidx-1):
    print('-------------------------')
    print(idx)
    # ---- solve problem ----
    resultFlag, x_opt, u_opt, del_opt, f_opt, t_opt = optObj.solveProblem(idx, x0)
    # ---- update initial state (simulation) ----
    u0 = u_opt[:, 0].elements()
    # x0 = x_opt[:,1].elements()
    x0 = (x0 + dyn.f(x0, u0)*dt).elements()
    # ---- store values for plotting ----
    comp_time[idx] = t_opt
    success[idx] = resultFlag
    cost_plot[idx] = f_opt
    X_plot[:, idx+1] = x0
    U_plot[:, idx] = u0
    X_future[:, :, idx] = np.array(x_opt)
    if dyn.Nz > 0:
        del_plot[:, idx] = del_opt[0]
#  -------------------------------------------------------------------
# show sparsity pattern
# sliding_pack.plots.plot_sparsity(cs.vertcat(*opt.g), cs.vertcat(*opt.x), xu_opt)
#  -------------------------------------------------------------------

# Animation
#  -------------------------------------------------------------------
if show_anim:
#  -------------------------------------------------------------------
    fig, ax = sliding_pack.plots.plot_nominal_traj(x0_nom, x1_nom)
    fig.set_size_inches(8, 6, forward=True)
    # get slider and pusher patches
    dyn.set_patches(ax, X_plot)
    # call the animation
    ani = animation.FuncAnimation(
            fig,
            dyn.animate,
            fargs=(ax, X_plot, X_future),
            frames=Nidx-1,
            interval=dt*1000,  # microseconds
            blit=True,
            repeat=False,
    )
    ## to save animation, uncomment the line below:
    # ani.save('MPC_NLP_eight.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(5, 2, sharex=True)
fig.set_size_inches(12, 15, forward=True)
t_N_x = np.linspace(0, T, N)
t_N_u = np.linspace(0, T, N-1)
t_idx_x = t_N_x[0:Nidx]
t_idx_u = t_N_x[0:Nidx-1]
ctrl_g_idx = dyn.g_u.map(Nidx-1)
ctrl_g_val = ctrl_g_idx(U_plot, np.zeros((1, Nidx-1)))
#  -------------------------------------------------------------------
axs[0,0].plot(t_N_x, X_nom_val[0,0:N].T, color='b', label='nom')
axs[0,0].plot(t_idx_x, X_plot[0,:], color='g', linestyle='--', label='opt')
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles, labels)
axs[0,0].set_ylabel('x0')
axs[0,0].grid()
#  -------------------------------------------------------------------
axs[1,0].plot(t_N_x, X_nom_val[1,0:N].T, color='b', label='nom')
axs[1,0].plot(t_idx_x, X_plot[1,:], color='g', linestyle='--', label='opt')
handles, labels = axs[1,0].get_legend_handles_labels()
axs[1,0].legend(handles, labels)
axs[1,0].set_ylabel('x1')
axs[1,0].grid()
#  -------------------------------------------------------------------
axs[2,0].plot(t_N_x, X_nom_val[2,0:N].T*(180/np.pi), color='b', label='nom')
axs[2,0].plot(t_idx_x, X_plot[2,:]*(180/np.pi), color='g', linestyle='--', label='opt')
handles, labels = axs[2,0].get_legend_handles_labels()
axs[2,0].legend(handles, labels)
axs[2,0].set_ylabel('x2')
axs[2,0].grid()
#  -------------------------------------------------------------------
axs[3,0].plot(t_N_x, X_nom_val[3,0:N].T*(180/np.pi), color='b', label='nom')
axs[3,0].plot(t_idx_x, X_plot[3,:]*(180/np.pi), color='g', linestyle='--', label='opt')
handles, labels = axs[3,0].get_legend_handles_labels()
axs[3,0].legend(handles, labels)
axs[3,0].set_ylabel('x3')
axs[3,0].grid()
#  -------------------------------------------------------------------
axs[4,0].plot(t_idx_u, ctrl_g_val[0,:].T, color='b', label='left')
axs[4,0].plot(t_idx_u, ctrl_g_val[1,:].T, color='g', label='right')
# axs[4,0].plot(t_idx_u, (U_plot[2,:].T)*0.01, color='r', label='psi_dot')
handles, labels = axs[4,0].get_legend_handles_labels()
axs[4,0].legend(handles, labels)
axs[4,0].set_xlabel('time [s]')
axs[4,0].set_ylabel('fric cone')
axs[4,0].grid()
#  -------------------------------------------------------------------
axs[0,1].plot(t_idx_u, U_plot[0,:], color='g', linestyle='--', label='opt')
handles, labels = axs[0,1].get_legend_handles_labels()
axs[0,1].legend(handles, labels)
axs[0,1].set_ylabel('u0')
axs[0,1].grid()
#  -------------------------------------------------------------------
axs[1,1].plot(t_idx_u, U_plot[1,:], color='g', linestyle='--', label='opt')
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].legend(handles, labels)
axs[1,1].set_ylabel('u1')
axs[1,1].grid()
#  -------------------------------------------------------------------
# axs[2,1].plot(t_idx_u, U_plot[2,:]*(180/np.pi), color='g', linestyle='--', label='opt+')
# axs[2,1].plot(t_idx_u, U_plot[3,:]*(180/np.pi), color='g', linestyle='--', label='opt-')
# handles, labels = axs[2,1].get_legend_handles_labels()
# axs[2,1].legend(handles, labels)
# axs[2,1].set_ylabel('u2')
# axs[2,1].grid()
#  -------------------------------------------------------------------
axs[3,1].plot(t_idx_u, comp_time, color='b')
handles, labels = axs[3,1].get_legend_handles_labels()
axs[3,1].legend(handles, labels)
axs[3,1].set_ylabel('time [s]')
axs[3,1].grid()
#  -------------------------------------------------------------------
# axs[4,1].plot(t_idx_u, success, color='r', label='succ')
axs[4,1].plot(t_idx_u, cost_plot, color='b', label='cost')
handles, labels = axs[4,1].get_legend_handles_labels()
axs[4,1].legend(handles, labels)
axs[4,1].set_xlabel('time [s]')
axs[4,1].set_ylabel('u')
axs[4,1].grid()
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
