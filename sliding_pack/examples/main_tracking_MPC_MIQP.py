## Author: Joao Moura
## Date: 21/08/2020
#  -------------------------------------------------------------------
## Description:
#  This script implements a quadratic programming (QP) optimal controller (OC)
#  for tracking a line trajectory of a square slider object with a single
#  and fixed contacter pusher.
#  -------------------------------------------------------------------

## Import Libraries
#  -------------------------------------------------------------------
import os
import sys
import time
import numpy as np
import casadi as cs
#import numpy.matlib as nplib
from scipy.integrate import dblquad 
#from sys import path
#path.append(r"/Users/joaomoura/local/casadi")
# import casadi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.transforms as transforms
#  -------------------------------------------------------------------
import sliding_pack
#  -------------------------------------------------------------------

## Set Problem constants
#  -------------------------------------------------------------------
N_x = 4 # number of state variables
N_u = 3 # number of actions variables
N_i = 3 # number of integer variables
a = 0.09 # side dimension of the square slider in meters
miu_p = 0.2 # coefficient of friction between pusher and slider
T = 12 # time of the simulation is seconds
freq = 25 # number of increments per second
r_pusher = 0.01 # radius of the cylindrical pusher in meter
# Mm = np.array([1, 5, 5, 5, 5, 5, 5, 4]) # mode scheduling
# Mm = np.ones((15,), dtype=int)
# Mm = np.array([1, 2, 4, 8, 16, 32]) # mode scheduling
# Mm = np.array([1, 2, 4, 8]) # mode scheduling
# Mm = np.array([1, 5, 5, 4]) # mode scheduling
Mm = np.array([1, 4, 4, 3]) # mode scheduling
bigM = 500 # big M for the Mixed Integer optimization
epsilon = 0.0
f_lim = 0.3 # limit on the actuations
psi_dot_lim = 3.0 # limit on the actuations
psi_lim = 40*(np.pi/180.0)
x_init_val = [-0.01, 0.03, 30*(np.pi/180.), 0*(np.pi/180.)]
u_init_val = [0.0, 0.0, 0.0]
solver_name = 'gurobi'
opts_dict = {'print_time': 0}
no_printing = True
code_gen = False
show_anim = True
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N_xu = N_x + N_u # number of optimization variables
N_z = N_i*Mm.size
N_MPC = np.sum(Mm) # time horizon for the MPC controller
N_m = Mm.size
N = T*freq # total number of iterations
Nidx = int(N)
# Nidx = 3
NN = N + N_MPC # total number of steps
dt = 1.0/freq # sampling time
N_var = (N_xu)*N_MPC
# Area integral of norm of the distance for a square:
int_square = lambda a: dblquad(lambda x,y: np.sqrt(x**2 + y**2), -a/2, a/2, -a/2, a/2)[0]
int_A = int_square(a)
#  -------------------------------------------------------------------
## get string name
prog_name = 'MPC' + '_TH' + str(N_MPC) + '_' + solver_name + '_codeGen_' + str(code_gen)
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
dx = cs.SX.sym('dx', 4)
# u - control vector
# u[0] - normal force in the local frame
# u[1] - tangential force in the local frame
# u[2] - relative sliding velocity between pusher and slider
u = cs.SX.sym('u', N_u)
# b - dynamic parameters
# b[0] - slider lenght [m]
# b[1] - radious of the pusher [m]
beta = [a, r_pusher]
# z - modes
# z[0] - Sticking mode
# z[1] - Sliding up mode
# z[2] - Sliding down mode
z = cs.SX.sym('x', N_i)
#  ------------------------------------------------------------------

## Build Motion Model
#  -------------------------------------------------------------------
R_pusher_func = sliding_pack.dyn.square_slider_quasi_static_ellipsoidal_limit_surface_R
#  -------------------------------------------------------------------
p_pusher_func = cs.Function('p_pusher_func', [x], [sliding_pack.dyn.square_slider_quasi_static_ellipsoidal_limit_surface_p(x, beta)], ['x'], ['p'])
#  -------------------------------------------------------------------
f_func = cs.Function('f_func', [x,u], [sliding_pack.dyn.square_slider_quasi_static_ellipsoidal_limit_surface_f(x,u, beta)],['x','u'],['xdot'])
#  -------------------------------------------------------------------

## Compute Jacobians
#  -------------------------------------------------------------------
A_func = cs.Function('A_func', [x,u], [cs.jacobian(f_func(x,u), x)], ['x', 'u'], ['A'])
B_func = cs.Function('B_func', [x,u], [cs.jacobian(f_func(x,u), u)], ['x', 'u'], ['B'])
#  -------------------------------------------------------------------

## Define constraint functions
#  -------------------------------------------------------------------
## ---- Input variables ---
x_bar = cs.SX.sym('x_bar', N_x)
x_bar_next = cs.SX.sym('x_bar_next', N_x)
u_bar = cs.SX.sym('u_bar', N_u)
## ---- Define Dynamic constraints ----
dyn_err_f = cs.Function('dyn_err_f', [x, u, x_bar, x_bar_next, u_bar], 
        [x_bar_next-x_bar-dt*(cs.mtimes(A_func(x,u), x_bar) + cs.mtimes(B_func(x,u),u_bar))])
## ---- Define Control constraints ----
fric_cone_c = cs.Function('fric_cone_c', [u], [cs.vertcat(miu_p*u[0]+u[1], miu_p*u[0]-u[1])])
fric_cone_C = fric_cone_c.map(NN-1)
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
X_nom_val, dX_nom_val = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  ------------------------------------------------------------------
# control path variables
u_nom = cs.SX.sym('u_nom', N_u, NN-1)
#  ------------------------------------------------------------------
# declare cost function
W_f = cs.diag(cs.SX([1.0,1.0,0.01,0.0]))
vel_error = dx - f_func(x, u)
cost_f = cs.Function('cost', [x, dx, u], [cs.dot(vel_error,cs.mtimes(W_f,vel_error))])
cost_F = cost_f.map(NN-1)
#  -------------------------------------------------------------------
opt = sliding_pack.opt.OptVars()
# define cost function
opt.f = cs.sum2(cost_F(X_nom_val[:,0:-1], dX_nom_val, u_nom))
# define optimization variables
opt.x = cs.vertcat(*u_nom.elements())
# define Sticking constraint
opt.g = cs.horzcat(*fric_cone_C(u_nom).elements())
#  -------------------------------------------------------------------
# Generating solver
prob = {'f': opt.f, 'x': opt.x, 'g':opt.g}
solver = cs.nlpsol('solver', 'ipopt', prob)
#  -------------------------------------------------------------------
# Instanciating optimizer arguments
args = sliding_pack.opt.OptArgs()
# initial condition for opt var
args.x0 = [0.0]*((NN-1)*N_u)
# opt var boundaries
args.lbx = [0.0, -cs.inf, 0.0]*(NN-1)
args.ubx = [cs.inf, cs.inf, 0.0]*(NN-1)
# arg for sticking constraint
args.lbg = [0.0]*((NN-1)*2)
args.ubg = [cs.inf]*((NN-1)*2)
#  -------------------------------------------------------------------
# Solve optimization problem
sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg)
u_sol = sol['x']
U_nom_val = cs.horzcat(u_sol[0::N_u],u_sol[1::N_u],u_sol[2::N_u]).T
#  -------------------------------------------------------------------

## Define variables for optimization
#  -------------------------------------------------------------------
## ---- Input variables ---
X_bar = cs.SX.sym('x_bar', N_x, N_MPC)
U_bar = cs.SX.sym('u_bar', N_u, N_MPC-1)
## ---- Nominal state and action variables ----
X_nom = cs.SX.sym('x_nom', N_x, N_MPC)
U_nom = cs.SX.sym('u_nom', N_u, N_MPC-1)
X = X_bar + X_nom
U = U_bar + U_nom
## ---- Initial state and action variables ----
x_init = cs.SX.sym('x0', N_x)
u_init = cs.SX.sym('u0', N_u)
## ---- discrete variables ----
Zm = cs.SX.sym('z', N_i, N_m)
Z = cs.repmat(Zm[:, 0], 1, Mm[0])
for i in range(1, N_m):
    Z = cs.horzcat(Z, cs.repmat(Zm[:, i], 1, Mm[i]))
Z_val = cs.SX.zeros(3,1)
Z_val[0] = 1
Zm0 = cs.DM(cs.repmat(Z_val, 1, N_m))
#  -------------------------------------------------------------------

## Set up QP Optimization Problem
#  -------------------------------------------------------------------
opt = sliding_pack.opt.OptVars()
args = sliding_pack.opt.OptArgs()
## ---- Set optimization objective ----------
Qcost = cs.diag(cs.SX([1.0,1.0,0.01,0.0])); QcostN = Qcost
# Rcost = 0.1*cs.diag(cs.SX([1.0,1.0,0.0]))
Rcost = cs.diag(cs.SX([0.0,0.0,0.0]))
wcost = cs.SX([0.0, 0.3, 0.1, 0.1, 0.1, 0.1])
Q = cs.SX.sym('Q', N_x, N_x)
cost = cs.Function('cost', [Q, x, u], [cs.dot(x,cs.mtimes(Q,x)) + cs.dot(u,cs.mtimes(Rcost,u))])
cost_f = cs.Function('cost_f', [x, u], [cost(Qcost, x, u)])
cost_F = cost_f.map(N_MPC-1)
w_zi = cs.SX.sym('w_zi') # weight of the mode i
Nmi = cs.SX.sym('Nmi') # number of samples for mode i
cost_z = cs.Function('cost_z', [w_zi, Nmi, z], [Nmi*cs.dot(z,w_zi*z)])
cost_Z = cost_z.map(N_m)
## ---- cost function ----
opt.f = cs.sum2(cost_F(X_bar[:,0:-1], U_bar))
opt.f += cost(QcostN, X_bar[:,-1], cs.SX(N_u, 1)) 
# opt.f += cs.sum2(cost_Z(wcost, Mm, Zm))
## ---- Set optimization variables ----
opt.x = []
args.x0 = []
args.lbx = []
args.ubx = []
opt.discrete = []
for i in range(N_MPC-1):
    ## ---- Add States to optimization variables ---
    opt.x += X_bar[:,i].elements()
    args.x0 += X_nom_val[:,i].elements()
    args.lbx += [-cs.inf]*(N_x-1)
    args.ubx += [cs.inf]*(N_x-1)
    args.lbx += [-psi_lim]
    args.ubx += [psi_lim]
    opt.discrete += [False]*N_x
    ## ---- Add Actions to optimization variables ---
    opt.x += U_bar[:,i].elements()
    args.x0 += U_nom_val[:,i].elements()
    args.lbx += [-cs.inf]*N_u
    args.ubx += [cs.inf]*N_u
    opt.discrete += [False]*N_u
opt.x += X_bar[:,-1].elements()
args.x0 += X_nom_val[:,-1].elements()
args.lbx += [-cs.inf]*(N_x-1)
args.ubx += [cs.inf]*(N_x-1)
args.lbx += [-psi_lim]
args.ubx += [psi_lim]
opt.discrete += [False]*N_x
for i in range(N_m):
    opt.x += Zm[:,i].elements()
    args.x0 += Zm0[:,i].elements()
    args.lbx += [0]*N_u
    args.ubx += [1]*N_u
    opt.discrete += [True]*N_i
## ---- Set optimzation constraints ----
# opt.g = []
opt.g = [X[:,0]-x_init] ## Initial Conditions
args.lbg = [0]*N_x
args.ubg = [0]*N_x
for i in range(N_MPC-1):
    ## ---- Dynamic constraints ---- 
    opt.g += dyn_err_f(X_nom[:,i], U_nom[:,i], X_bar[:,i], X_bar[:,i+1], U_bar[:,i]).elements()
    args.lbg += [0]*N_x
    args.ubg += [0]*N_x
    ## ---- Friction cone constraints ----
    opt.g += (fric_cone_c(U[:,i]) + bigM*cs.vertcat(Z[1,i], Z[2,i])).elements()
    args.lbg += [0.0]*2
    args.ubg += [cs.inf]*2
    opt.g += (fric_cone_c(U[:,i]) - bigM*cs.vertcat(1-Z[2,i], 1-Z[1,i])).elements()
    args.lbg += [-cs.inf]*2
    args.ubg += [0.0]*2
    opt.g += [U_bar[2,i] + bigM*Z[2,i] + epsilon*Z[0,i]]
    opt.g += [U_bar[2,i] - bigM*Z[1,i] - epsilon*Z[0,i]]
    args.lbg += [epsilon, -cs.inf]
    args.ubg += [cs.inf,  -epsilon]
    ## ---- Action Constraints ---- 
    # [normal vel, tangential vel, relative sliding vel]
    opt.g += U[:,i].elements()
    args.lbg += [0.0,  -f_lim, -psi_dot_lim]
    args.ubg += [f_lim, f_lim,  psi_dot_lim]
for i in range(N_m):
    ## Integer summation
    opt.g += [cs.sum1(Zm[:,i])]
    args.lbg += [1.0]
    args.ubg += [1.0]
## ---- Set optimization parameters ----
opt.p = []
opt.p += x_init.elements()
opt.p += u_init.elements()
opt.p += X_nom.elements()
opt.p += U_nom.elements()
## ---- Set solver options ----
opts_dict['discrete'] = opt.discrete # add integer variables
if solver_name == 'gurobi':
    if no_printing: opts_dict['gurobi.OutputFlag'] = 0
## ---- Create solver ----
prob = {'f': opt.f, 'x': cs.vertcat(*opt.x), 'g': cs.vertcat(*opt.g), 'p': cs.vertcat(*opt.p)}
if (solver_name == 'gurobi'):
    solver = cs.qpsol('solver', solver_name, prob, opts_dict)
#  -------------------------------------------------------------------

## Initialize variables for plotting
#  -------------------------------------------------------------------
X_plot = np.empty([N_x, Nidx])
U_plot = np.empty([N_u, Nidx-1])
Z_plot = np.empty([N_i, Nidx])
X_plot[:,0] = x_init_val
X_future = np.empty([N_x, N_MPC, Nidx])
comp_time = np.empty(Nidx-1)
success = np.empty(Nidx-1)
#  -------------------------------------------------------------------

## Set arguments and solve
#  -------------------------------------------------------------------
x0 = x_init_val
u0 = u_init_val
for idx in range(Nidx-1):
    ## setting parameters
    args.p = []
    args.p += x0
    args.p += u0
    args.p += X_nom_val[:,idx:(idx+N_MPC)].elements()
    args.p += U_nom_val[:,idx:(idx+N_MPC-1)].elements()
    ## ---- Solve the optimization ----
    start_time = time.time()
    sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg, p=args.p)
    stats = solver.stats()
    if stats['success'] == True:
        success[idx] = 1
    else:
        success[idx] = 0
    xz_opt = sol['x']
    x_opt = xz_opt[0:(N_MPC*N_xu-N_u)]
    z_opt = xz_opt[(N_MPC*N_xu-N_u):(N_MPC*N_xu-N_u+N_z)]
    x_i = sol['x'][(N_MPC*N_xu-N_u):]
    # ---- save computation time ---- 
    comp_time[idx] = time.time() - start_time
    ## ---- Compute actual trajectory and controls ----
    X_bar_opt = cs.horzcat(x_opt[0::N_xu],x_opt[1::N_xu],x_opt[2::N_xu],x_opt[3::N_xu]).T
    U_bar_opt = cs.horzcat(x_opt[4::N_xu],x_opt[5::N_xu],x_opt[6::N_xu]).T
    X_bar_opt = X_bar_opt
    U_bar_opt = U_bar_opt
    X_opt = X_bar_opt + X_nom_val[:,idx:(idx+N_MPC)]
    U_opt = U_bar_opt + U_nom_val[:,idx:(idx+N_MPC-1)]
    ## ---- Update initial conditions and warm start ----
    u0 = U_opt[:,0].elements()
    # x0 = X_opt[:,1].elements()
    x0 = (x0 + f_func(x0, u0)*dt).elements()
    # print(x0[N_x-1]*(180/np.pi))
    ## ---- Store values for plotting ----
    X_plot[:,idx+1] = x0
    U_plot[:,idx] = u0
    Z_plot[:,idx] = z_opt[0:N_i].T
    X_future[:,:,idx] = np.array(X_opt)
    # ---- warm start ---- 
    # x_opt[0::N_xu] = [0.0]*(N_MPC)
    # x_opt[1::N_xu] = [0.0]*(N_MPC)
    # x_opt[2::N_xu] = [0.0]*(N_MPC)
    # x_opt[3::N_xu] = [0.0]*(N_MPC)
    # x_opt[4::N_xu] = [0.0]*(N_MPC-1)
    # x_opt[5::N_xu] = [0.0]*(N_MPC-1)
    # x_opt[6::N_xu] = [0.0]*(N_MPC-1)
    args.x0 = cs.vertcat(x_opt,z_opt).elements()
#  -------------------------------------------------------------------
# show sparsity pattern
# sliding_pack.plots.plot_sparsity(cs.vertcat(*opt.g), cs.vertcat(*opt.x), xz_opt)
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(6, 2, sharex=True, figsize=(12,8))
t_N_x = np.linspace(0, T, N)
t_N_u = np.linspace(0, T, N-1)
t_idx_x = t_N_x[0:Nidx]
t_idx_u = t_N_x[0:Nidx-1]
X_nom_val = np.array(X_nom_val)
fric_cone_val = fric_cone_idx(U_plot)
#  -------------------------------------------------------------------
axs[0,0].plot(t_N_x, X_nom_val[0,0:N], color='b', label='nom')
axs[0,0].plot(t_idx_x, X_plot[0,:], color='g', linestyle='--', label='opt')
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles, labels)
axs[0,0].set_ylabel('x0')
axs[0,0].grid()
#  -------------------------------------------------------------------
axs[1,0].plot(t_N_x, X_nom_val[1,0:N], color='b', label='nom')
axs[1,0].plot(t_idx_x, X_plot[1,:], color='g', linestyle='--', label='opt')
handles, labels = axs[1,0].get_legend_handles_labels()
axs[1,0].legend(handles, labels)
axs[1,0].set_ylabel('x1')
axs[1,0].grid()
#  -------------------------------------------------------------------
axs[2,0].plot(t_N_x, X_nom_val[2,0:N], color='b', label='nom')
axs[2,0].plot(t_idx_x, X_plot[2,:], color='g', linestyle='--', label='opt')
handles, labels = axs[2,0].get_legend_handles_labels()
axs[2,0].legend(handles, labels)
axs[2,0].set_ylabel('x2')
axs[2,0].grid()
#  -------------------------------------------------------------------
axs[3,0].plot(t_N_x, X_nom_val[3,0:N]*(180./np.pi), color='b', label='nom')
axs[3,0].plot(t_idx_x, X_plot[3,:]*(180./np.pi), color='g', linestyle='--', label='opt')
handles, labels = axs[3,0].get_legend_handles_labels()
axs[3,0].legend(handles, labels)
axs[3,0].set_ylabel('x3')
axs[3,0].grid()
#  -------------------------------------------------------------------
axs[4,0].plot(t_idx_u, fric_cone_val[1,:].T, color='b', label='up')
axs[4,0].plot(t_idx_u, fric_cone_val[0,:].T, color='g', label='down')
axs[4,0].plot(t_idx_u, (U_plot[2,:].T)*0.01, color='r', label='psi_dot')
handles, labels = axs[4,0].get_legend_handles_labels()
axs[4,0].legend(handles, labels)
axs[4,0].set_xlabel('time [s]')
axs[4,0].set_ylabel('fric cone')
axs[4,0].grid()
#  -------------------------------------------------------------------
axs[0,1].plot(t_N_u, U_nom_val[0,0:N-1].T, color='b', label='nom')
axs[0,1].plot(t_idx_u, U_plot[0,:], color='g', linestyle='--', label='opt')
handles, labels = axs[0,1].get_legend_handles_labels()
axs[0,1].legend(handles, labels)
axs[0,1].set_ylabel('u0')
axs[0,1].grid()
#  -------------------------------------------------------------------
axs[1,1].plot(t_N_u, U_nom_val[1,0:N-1].T, color='b', label='nom')
axs[1,1].plot(t_idx_u, U_plot[1,:], color='g', linestyle='--', label='opt')
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].legend(handles, labels)
axs[1,1].set_ylabel('u1')
axs[1,1].grid()
#  -------------------------------------------------------------------
axs[2,1].plot(t_N_u, U_nom_val[2,0:N-1].T, color='b', label='nom')
axs[2,1].plot(t_idx_u, U_plot[2,:], color='g', linestyle='--', label='opt')
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
axs[4,1].plot(t_idx_x, Z_plot[0,:], color='r', label='sticky')
axs[4,1].plot(t_idx_x, Z_plot[1,:], color='b', label='up')
axs[4,1].plot(t_idx_x, Z_plot[2,:], color='g', label='down')
handles, labels = axs[4,1].get_legend_handles_labels()
axs[4,1].legend(handles, labels)
axs[4,1].set_ylabel('modes')
axs[4,1].grid()
#  -------------------------------------------------------------------
axs[5,1].plot(t_idx_u, (U_plot[1,:]*U_plot[2,:]).T, color='g', label='u1*u2')
axs[5,1].plot(t_idx_u, 0.01*success, color='r', label='succ')
handles, labels = axs[5,1].get_legend_handles_labels()
axs[5,1].legend(handles, labels)
axs[5,1].set_xlabel('time [s]')
axs[5,1].set_ylabel('u1*u2')
axs[5,1].grid()
#  -------------------------------------------------------------------

# Animation
#  -------------------------------------------------------------------
if show_anim:
#  -------------------------------------------------------------------
    fig, ax = sliding_pack.plots.plot_nominal_traj(x0_nom[0:N], x1_nom[0:N])
    # get slider and pusher patches
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
            repeat=False)
    ## to save animation, uncomment the line below:
    # ani.save('MPC_MIQP_eight.mp4', fps=25, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
