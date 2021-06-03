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
import os
import sys
import time
import numpy as np
import casadi as cs
# import casadi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

# Define system dynamics
#  -------------------------------------------------------------------
dyn = sliding_pack.dyn.System_square_slider_quasi_static_ellipsoidal_limit_surface(
        slider_dim=0.09, pusher_radious=0.01)
#  -------------------------------------------------------------------

## Set Problem constants
#  -------------------------------------------------------------------
N_s = 1
N_xu = dyn.Nx + dyn.Nu + N_s # number of optimization variables
a = 0.09 # side dimension of the square slider in meters
miu_p = 0.2 # coefficient of friction between pusher and slider
W_f = cs.diag(cs.SX([1.0,1.0,0.01,0.0]))
T = 12 # time of the simulation is seconds
freq = 50 # number of increments per second
r_pusher = 0.01 # radius of the cylindrical pusher in meter
# N_MPC = 150 # time horizon for the MPC controller
# N_MPC = 63 # time horizon for the MPC controller
N_MPC = 15 # time horizon for the MPC controller
# N_MPC = 10 # time horizon for the MPC controller
x_init_val = [-0.01, 0.03, 30*(np.pi/180.), 0]
u_init_val = [0.0, 0.0, 0.0, 0.0]
f_lim = 0.3 # limit on the actuations
psi_dot_lim = 3.0 # limit on the actuations
psi_lim = 40*(np.pi/180.0)
# solver_name = 'ipopt'
solver_name = 'snopt'
# solver_name = 'gurobi'
# solver_name = 'qpoases'
opts_dict = {'print_time': 0}
no_printing = True
code_gen = False
show_anim = True
#  -------------------------------------------------------------------
## get string name
prog_name = 'MPC' + '_TH' + str(N_MPC) + '_' + solver_name + '_codeGen_' + str(code_gen)
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
dt = 1.0/freq # sampling time
N = int(T*freq) # total number of iterations
T_MPC = N_MPC*dt
# NN = N + N_MPC # total number of steps
Nidx = int(N)
# Nidx = 3
#  -------------------------------------------------------------------

## Define constraint functions
#  -------------------------------------------------------------------
## ---- Input variables ---
x_next = cs.SX.sym('x_next', dyn.Nx)
## ---- Define Dynamic constraints ----
dyn_err_f = cs.Function('dyn_err_f', [dyn.x, dyn.u, x_next], 
        [x_next-dyn.x-dt*dyn.f(dyn.x,dyn.u)])
## ---- Define Control constraints ----
fric_cone_c = cs.Function('fric_cone_c', [dyn.u], [cs.vertcat(miu_p*dyn.u[0]+dyn.u[1], miu_p*dyn.u[0]-dyn.u[1])])
fric_cone_idx = fric_cone_c.map(Nidx-1)
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

## Define variables for optimization
#  -------------------------------------------------------------------
## ---- Input variables ---
X = cs.SX.sym('X', dyn.Nx, N_MPC)
U = cs.SX.sym('U', dyn.Nu, N_MPC-1)
X_nom = cs.SX.sym('X_nom', dyn.Nx, N_MPC)
## ---- Initial state and action variables ----
x_init = cs.SX.sym('x0', dyn.Nx)
u_init = cs.SX.sym('u0', dyn.Nu)
x_nom = cs.SX.sym('x_nom', dyn.Nx)
## ---- Slack variable for complementarity constraint ----
del_cc = cs.SX.sym('del_cc', N_MPC-1)
#  -------------------------------------------------------------------

## Set up QP Optimization Problem
#  -------------------------------------------------------------------
## ---- Initialize optimization and argument variables ---
opt = sliding_pack.opt.OptVars()
args = sliding_pack.opt.OptArgs()
## ---- Define optimization objective ----------
pos_err = dyn.x - x_nom
cost_f = cs.Function('cost_f', [dyn.x, x_nom], [cs.dot(pos_err,cs.mtimes(W_f,pos_err))])
cost_F = cost_f.map(N_MPC)
opt.f = cs.sum2(cost_F(X, X_nom))
Ks_max = 50.0; Ks_min = 0.1; xs = np.linspace(0,1,N_MPC-1)
Ks = Ks_max*cs.exp(xs*cs.log(Ks_min/Ks_max))
opt.f += cs.sum1(Ks*(del_cc**2))
## ---- Set optimization variables ----
opt.x = []
args.x0 = []
args.lbx = []
args.ubx = []
for i in range(N_MPC-1):
    ## ---- Add States to optimization variables ---
    opt.x += X[:,i].elements()
    args.x0 += X_nom_val[:,i].elements()
    args.lbx += [-cs.inf]*(dyn.Nx-1)
    args.ubx += [cs.inf]*(dyn.Nx-1)
    args.lbx += [-psi_lim]
    args.ubx += [psi_lim]
    ## ---- Add Actions to optimization variables ---
    opt.x += U[:,i].elements()
    args.lbx += [0.0,  -f_lim,         0.0,         0.0]
    args.ubx += [f_lim, f_lim, psi_dot_lim, psi_dot_lim]
    args.x0 += [0.0,     0.0, 0.0, 0.0]
    ## ---- Add slack variables ---
    opt.x += del_cc[i].elements()
    args.x0 += [0.0]
    args.lbx += [-cs.inf]
    args.ubx += [cs.inf]
opt.x += X[:,-1].elements()
args.x0 += X_nom_val[:,-1].elements()
args.lbx += [-cs.inf]*(dyn.Nx-1)
args.ubx += [cs.inf]*(dyn.Nx-1)
args.lbx += [-psi_lim]
args.ubx += [psi_lim]
## ---- Set optimzation constraints ----
opt.g = (X[:,0]-x_init).elements() ## Initial Conditions
args.lbg = [0.0]*dyn.Nx
args.ubg = [0.0]*dyn.Nx
for i in range(N_MPC-1):
    ## ---- Dynamic constraints ---- 
    opt.g += dyn_err_f(X[:,i], U[:,i], X[:,i+1]).elements()
    args.lbg += [0]*dyn.Nx
    args.ubg += [0]*dyn.Nx
    ## ---- Friction cone constraints ----
    opt.g += fric_cone_c(U[:,i]).elements()
    args.lbg += [0.0]*2
    args.ubg += [cs.inf]*2
    ## Complementary constraint
    opt.g += [(miu_p*U[0,i]-U[1,i])*U[3,i]+(miu_p*U[0,i]+U[1,i])*U[2,i] + del_cc[i]]
    # opt.g += [(miu_p*U[0,i]-U[1,i])*U[3,i]+(miu_p*U[0,i]+U[1,i])*U[2,i]]
    args.lbg += [0.0]
    args.ubg += [0.0]
## ---- Set optimization parameters ----
opt.p = []
opt.p += x_init.elements()
opt.p += X_nom.elements()
## ---- Set solver options ----
if solver_name == 'ipopt':
    if no_printing: opts_dict['ipopt.print_level'] = 0
    opts_dict['ipopt.jac_d_constant'] = 'yes'
    opts_dict['ipopt.warm_start_init_point'] = 'yes'
    opts_dict['ipopt.hessian_constant'] = 'yes'
if solver_name == 'snopt':
    if no_printing: opts_dict['snopt'] = {'Major print level': '0', 'Minor print level': '0'}
    opts_dict['snopt']['Hessian updates'] = 1
if solver_name == 'qpoases':
    if no_printing: opts_dict['printLevel'] = 'none'
    opts_dict['sparse'] = True
if solver_name == 'gurobi':
    if no_printing: opts_dict['gurobi.OutputFlag'] = 0
## ---- Create solver ----
prob = {'f': opt.f, 'x': cs.vertcat(*opt.x), 'g': cs.vertcat(*opt.g), 'p': cs.vertcat(*opt.p)}
if (solver_name == 'ipopt') or (solver_name == 'snopt'):
    solver = cs.nlpsol('solver', solver_name, prob, opts_dict)
    if code_gen:
        if not os.path.isfile('./' + prog_name + '.so'):
            solver.generate_dependencies(prog_name + '.c')
            os.system('gcc -fPIC -shared -O3 ' + prog_name + '.c -o ' + prog_name + '.so')
        solver = cs.nlpsol('solver', solver_name, prog_name + '.so', opts_dict)
elif (solver_name == 'gurobi') or (solver_name == 'qpoases'):
    solver = cs.qpsol('solver', solver_name, prob, opts_dict)
#  -------------------------------------------------------------------

## Initialize variables for plotting
#  -------------------------------------------------------------------
X_plot = np.empty([dyn.Nx, Nidx])
U_plot = np.empty([dyn.Nu, Nidx-1])
del_plot = np.empty([1, Nidx-1])
X_plot[:,0] = x_init_val
X_future = np.empty([dyn.Nx, N_MPC, Nidx])
comp_time = np.empty(Nidx-1)
success = np.empty(Nidx-1)
cost_plot = np.empty(Nidx-1)
#  -------------------------------------------------------------------

## Set arguments and solve
#  -------------------------------------------------------------------
x0 = x_init_val
u0 = u_init_val
for idx in range(Nidx-1):
    ## ---- setting parameters ---- 
    args.p = [] # set to empty before reinitialize
    args.p += x0
    args.p += X_nom_val[:,idx:(idx+N_MPC)].elements()
    ## ---- Solve the optimization ----
    start_time = time.time()
    sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg, p=args.p)
    # ---- save computation time ---- 
    comp_time[idx] = time.time() - start_time
    print('--------------------------------')
    print(idx,'out of',Nidx-1)
    stats = solver.stats()
    if stats['success'] == True:
        success[idx] = 1
    else:
        success[idx] = 0
    cost_plot[idx] = sol['f']
    xu_opt = sol['x']
    ## ---- Compute actual trajectory and controls ----
    x_opt = cs.horzcat(xu_opt[0::N_xu],xu_opt[1::N_xu],xu_opt[2::N_xu],xu_opt[3::N_xu]).T
    u_opt = cs.horzcat(xu_opt[4::N_xu],xu_opt[5::N_xu],xu_opt[6::N_xu],xu_opt[7::N_xu]).T
    del_opt = x_opt[8::N_xu].elements()
    ## ---- Update initial conditions ----
    u0 = u_opt[:,0].elements()
    # x0 = x_opt[:,1].elements()
    x0 = (x0 + dyn.f(x0, u0)*dt).elements()
    ## ---- Store values for plotting ----
    X_plot[:,idx+1] = x0
    U_plot[:,idx] = u0
    X_future[:,:,idx] = np.array(x_opt)
    del_plot[:,idx] = del_opt[0]
    # ---- warm start ---- 
    # xu_opt[0::N_xu] = [0.0]*(N_MPC)
    # xu_opt[1::N_xu] = [0.0]*(N_MPC)
    # xu_opt[2::N_xu] = [0.0]*(N_MPC)
    # xu_opt[3::N_xu] = [0.0]*(N_MPC)
    # xu_opt[4::N_xu] = [0.0]*(N_MPC-1)
    # xu_opt[5::N_xu] = [0.0]*(N_MPC-1)
    xu_opt[6::N_xu] = [0.0]*(N_MPC-1)
    xu_opt[7::N_xu] = [0.0]*(N_MPC-1)
    xu_opt[8::N_xu] = [0.0]*(N_MPC-1)
    args.x0 = xu_opt.elements()
    # args.x0 = [0.0]*(len(args.x0))
#  -------------------------------------------------------------------
# show sparsity pattern
# sliding_pack.plots.plot_sparsity(cs.vertcat(*opt.g), cs.vertcat(*opt.x), xu_opt)
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(5, 2, sharex=True, figsize=(12,8))
t_N_x = np.linspace(0, T, N)
t_N_u = np.linspace(0, T, N-1)
t_idx_x = t_N_x[0:Nidx]
t_idx_u = t_N_x[0:Nidx-1]
fric_cone_val = fric_cone_idx(U_plot)
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
axs[4,0].plot(t_idx_u, fric_cone_val[0,:].T, color='b', label='left')
axs[4,0].plot(t_idx_u, fric_cone_val[1,:].T, color='g', label='right')
axs[4,0].plot(t_idx_u, (U_plot[2,:].T)*0.01, color='r', label='psi_dot')
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
axs[2,1].plot(t_idx_u, U_plot[2,:]*(180/np.pi), color='g', linestyle='--', label='opt+')
axs[2,1].plot(t_idx_u, U_plot[3,:]*(180/np.pi), color='g', linestyle='--', label='opt-')
handles, labels = axs[2,1].get_legend_handles_labels()
axs[2,1].legend(handles, labels)
axs[2,1].set_ylabel('u2')
axs[2,1].grid()
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

# Animation
#  -------------------------------------------------------------------
if show_anim:
#  -------------------------------------------------------------------
    fig, ax = sliding_pack.plots.plot_nominal_traj(x0_nom, x1_nom)
    # get slider and pusher patches
    x0 = np.array(X_plot[:,0].T)
    d0 = np.array(cs.mtimes(dyn.R(x0),[-a/2, -a/2, 0]).T)[0]
    slider, pusher, path_past, path_future = sliding_pack.plots.get_patches_for_square_slider_and_cicle_pusher(
            ax, 
            dyn.p, 
            dyn.R, 
            X_plot,
            a, r_pusher)
    # call the animation
    ani = animation.FuncAnimation(fig,
            sliding_pack.plots.animate_square_slider_and_circle_pusher,
            fargs=(slider, pusher, ax, dyn.p, dyn.R, X_plot, a, path_past, path_future, X_future),
            frames=Nidx-1,
            interval=dt*1000,
            blit=True,
            repeat=False,
    )
    ## to save animation, uncomment the line below:
    # ani.save('MPC_NLP_eight.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
