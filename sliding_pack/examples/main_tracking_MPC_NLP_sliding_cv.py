#!/usr/bin/python3

## Author: Joao Moura
## Date: 21/08/2020
#  -------------------------------------------------------------------
## Description:
#  This script implements a non-linear program (NLP) model predictive controller (MPC)
#  for tracking a trajectory of a square slider object with a single
#  and sliding contact pusher.
#  -------------------------------------------------------------------

# Import Libraries
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


# Add current directory to the LD_LIBRARY_PATH (for ubuntu)
#  -------------------------------------------------------------------
os.environ['LD_LIBRARY_PATH'] += os.getcwd()
#  -------------------------------------------------------------------

## Set Problem constants
#  -------------------------------------------------------------------
N_x = 4
N_u = 4
N_s = 2
N_xu = N_x + N_u + N_s # number of optimization variables
a = 0.09 # side dimension of the square slider in meters
miu_p = 0.25 # coefficient of friction between pusher and slider
W_f = cs.diag(cs.SX([1.0,1.0,0.01]))
T = 12 # time of the simulation is seconds
freq = 50 # number of increments per second
r_pusher = 0.01 # radius of the cylindrical pusher in meter
# N_MPC = 150 # time horizon for the MPC controller
# N_MPC = 63 # time horizon for the MPC controller
# N_MPC = 15 # time horizon for the MPC controller
N_MPC = 15 # time horizon for the MPC controller
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

## Define state and control vectors
#  -------------------------------------------------------------------
# x - state vector
# x[0] - x slider CoM position in the global frame
# x[1] - y slider CoM position in the global frame
# x[2] - slider orientation in the global frame
# x[3] - angle of pusher relative to slider
x = cs.SX.sym('x', N_x)
# dx - state vector derivative
dx = cs.SX.sym('dx', N_x)
# u - control vector
# u[0] - lambda +
# u[1] - lambda -
# u[2] - relative sliding velocity between pusher and slider up
# u[3] - relative sliding velocity between pusher and slider down
u = cs.SX.sym('u', N_u)
u_red_func = cs.Function('u_red_func', [u], [cs.vertcat((1./(2.0*miu_p))*(u[0]+u[1]), (1./2.0)*(u[0]-u[1]), u[2]-u[3])])
u_ = u_red_func(u)
# b - dynamic parameters
# b[0] - slider lenght [m]
# b[1] - radious of the pusher [m]
beta = [a, r_pusher]
# sl_cc - complementarity constraints slack variables
sl_cc = cs.SX.sym('sl_cc', 2)
#  -------------------------------------------------------------------

## Build Motion Model
#  -------------------------------------------------------------------
R_pusher_func = sliding_pack.dyn.square_slider_quasi_static_ellipsoidal_limit_surface_R
#  -------------------------------------------------------------------
p_pusher_func = cs.Function('p_pusher_func', [x], [sliding_pack.dyn.square_slider_quasi_static_ellipsoidal_limit_surface_p(x, beta)], ['x'], ['p'])
#  -------------------------------------------------------------------
f_func = cs.Function('f_func', [x,u], [sliding_pack.dyn.square_slider_quasi_static_ellipsoidal_limit_surface_f(x, u_, beta)],['x','u'],['xdot'])
#  -------------------------------------------------------------------

## Define constraint functions
#  -------------------------------------------------------------------
## ---- Input variables ---
x_next = cs.SX.sym('x_next', N_x)
## ---- Define Dynamic constraints ----
dyn_err_f = cs.Function('dyn_err_f', [x, u, x_next], 
        [x_next-x-dt*f_func(x,u)])
# dyn_err_F = dyn_err_f.map(N_MPC-1, 'serial', 2)
dyn_err_F = dyn_err_f.map(N_MPC-1, 'thread')
## ---- Define Control constraints ----
fric_cone_lim_c = cs.Function('fric_cone_lim_c', [u], [cs.vertcat(u[0]+u[1], u[0]-u[1])])
fric_cone_lim_C = fric_cone_lim_c.map(N_MPC-1, 'thread')
fric_cone_cc = cs.Function('fric_cone_lim_c', [u, sl_cc], [cs.vertcat(u[0]*u[2] + sl_cc[0], u[1]*u[3] + sl_cc[1])])
fric_cone_CC = fric_cone_cc.map(N_MPC-1, 'thread')
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
X = cs.SX.sym('X', N_x, N_MPC)
U = cs.SX.sym('U', N_u, N_MPC-1)
X_nom = cs.SX.sym('X_nom', N_x, N_MPC)
## ---- Initial state and action variables ----
x_init = cs.SX.sym('x0', N_x)
u_init = cs.SX.sym('u0', N_u)
x_nom = cs.SX.sym('x_nom', N_x)
## ---- Slack variable for complementarity constraint ----
SL_cc = cs.SX.sym('SL_cc', 2, N_MPC-1)
#  -------------------------------------------------------------------

## Set up QP Optimization Problem
#  -------------------------------------------------------------------
## ---- Initialize optimization and argument variables ---
opt = sliding_pack.opt.OptVars()
args = sliding_pack.opt.OptArgs()
## ---- Define optimization objective ----------
pos_err = x - x_nom
# cost_f = cs.Function('cost_f', [x, x_nom], [cs.dot(pos_err,cs.mtimes(W_f,pos_err))])
cost_f = cs.Function('cost_f', [x, x_nom], [cs.dot(x[0:(N_x-1)],cs.mtimes(W_f,x[0:(N_x-1)]))-2*cs.dot(x_nom[0:(N_x-1)],cs.mtimes(W_f,x[0:(N_x-1)]))])
cost_F = cost_f.map(N_MPC, 'thread')
opt.f = cs.sum2(cost_F(X, X_nom))
Ks_max = 50.0; Ks_min = 0.1; xs = np.linspace(0,1,N_MPC-1)
Ks = Ks_max*cs.exp(xs*cs.log(Ks_min/Ks_max))
opt.f += cs.sum1(Ks*(SL_cc[0,:].T**2))
opt.f += cs.sum1(Ks*(SL_cc[1,:].T**2))
## ---- Set optimization variables ----
opt.x = []
args.x0 = []
args.lbx = []
args.ubx = []
for i in range(N_MPC-1):
    ## ---- Add States to optimization variables ---
    opt.x += X[:,i].elements()
    args.x0 += X_nom_val[:,i].elements()
    args.lbx += [-cs.inf]*(N_x-1)
    args.ubx += [cs.inf]*(N_x-1)
    args.lbx += [-psi_lim]
    args.ubx += [psi_lim]
    ## ---- Add Actions to optimization variables ---
    opt.x += U[:,i].elements()
    args.lbx += [0.0,  0.0,         0.0,         0.0]
    args.ubx += [cs.inf, cs.inf, psi_dot_lim, psi_dot_lim]
    args.x0 += [0.0,     0.0, 0.0, 0.0]
    ## ---- Add slack variables ---
    opt.x += SL_cc[:,i].elements()
    args.x0 += [0.0, 0.0]
    args.lbx += [-cs.inf, -cs.inf]
    args.ubx += [cs.inf, cs.inf]
opt.x += X[:,-1].elements()
args.x0 += X_nom_val[:,-1].elements()
args.lbx += [-cs.inf]*(N_x-1)
args.ubx += [cs.inf]*(N_x-1)
args.lbx += [-psi_lim]
args.ubx += [psi_lim]
## ---- Set optimzation constraints ----
opt.g = (X[:,0]-x_init).elements() # Initial Conditions
args.lbg = [0.0]*N_x
args.ubg = [0.0]*N_x
## ---- Dynamic constraints ---- 
opt.g += dyn_err_F(X[:,:-1], U, X[:,1:]).elements()
args.lbg += [0.0]*(N_x*(N_MPC-1))
args.ubg += [0.0]*(N_x*(N_MPC-1))
## ---- Forces limit condition ----
opt.g += fric_cone_lim_C(U).elements()
args.lbg += [0.0, -2*f_lim]*(N_MPC-1)
args.ubg += [2*miu_p*f_lim, 2*f_lim]*(N_MPC-1)
# Complementary constraint
opt.g += fric_cone_CC(U, SL_cc).elements()
args.lbg += [0.0]*(2*(N_MPC-1))
args.ubg += [0.0]*(2*(N_MPC-1))
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
    # opts_dict['snopt']['Iterations limit'] = 200
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
            compile_instruc = 'gcc -fPIC -shared -O3 ' + prog_name + '.c -o ' + prog_name + '.so'
            flag = os.system(compile_instruc)
            if(flag==0): print("Compilation failed")
        # solver = cs.nlpsol('solver', solver_name, prog_name + '.so', opts_dict)
        solver = cs.nlpsol('solver', solver_name, prog_name + '.so', opts_dict)
elif (solver_name == 'gurobi') or (solver_name == 'qpoases'):
    solver = cs.qpsol('solver', solver_name, prob, opts_dict)
#  -------------------------------------------------------------------
# print(opts_dict)
# sys.exit()

## Initialize variables for plotting
#  -------------------------------------------------------------------
X_plot = np.empty([N_x, Nidx])
U_plot = np.empty([N_u, Nidx-1])
del_plot = np.empty([1, Nidx-1])
X_plot[:,0] = x_init_val
X_future = np.empty([N_x, N_MPC, Nidx])
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
    # print('--------------------------------')
    # print(idx,'out of',Nidx-1)
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
    del_opt = cs.horzcat(xu_opt[8::N_xu], xu_opt[9::N_xu]).T
    ## ---- Update initial conditions ----
    u0 = u_opt[:,0].elements()
    # x0 = x_opt[:,1].elements()
    x0 = (x0 + f_func(x0, u0)*dt).elements()
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
axs[4,0].plot(t_idx_u, U_plot[0,:], color='b', label='left')
axs[4,0].plot(t_idx_u, U_plot[1,:], color='g', label='right')
axs[4,0].plot(t_idx_u, (U_plot[2,:].T-U_plot[3,:].T)*0.01, color='r', label='psi_dot')
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
    d0 = np.array(cs.mtimes(R_pusher_func(x0),[-a/2, -a/2, 0]).T)[0]
    slider, pusher, path_past, path_future = sliding_pack.plots.get_patches_for_square_slider_and_cicle_pusher(
            ax, 
            p_pusher_func, 
            R_pusher_func, 
            X_plot,
            a, r_pusher)
    # call the animation
    ani = animation.FuncAnimation(fig,
            sliding_pack.plots.animate_square_slider_and_circle_pusher,
            fargs=(slider, pusher, ax, p_pusher_func, R_pusher_func, X_plot, a, path_past, path_future, X_future),
            frames=Nidx-1,
            interval=dt*1000,
            blit=True,
            repeat=False,
    )
    ## to save animation, uncomment the line below:
    # ani.save('MPC_NLP_circle.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
