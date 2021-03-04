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
import my_dynamics
import my_trajectories
import my_plots
import my_opt
#  -------------------------------------------------------------------

## Set Problem constants
#  -------------------------------------------------------------------
N_x = 4 # number of state variables
N_u = 3 # number of actions variables
N_i = 3 # number of integer variables
N_g = 10 # number of optimization constraints
g = 9.81 # gravity acceleration constant in meter per second square
a = 0.09 # side dimension of the square slider in meters
m = 0.827 # mass of the slider in kilo grams
miu_g = 0.35 # coeficient of friction between slider and table
miu_p = 0.1 # coeficient of friction between pusher and slider
T = 20 # time of the simulation is seconds
freq = 50 # numer of increments per second
r_pusher = 0.01 # radious of the cilindrical pusher in meter
Mm = np.array([1, 5, 5, 5, 5, 5, 5, 4]) # mode scheduling
bigM = 500 # big M for the Mixed Integer optimization
f_lim = 0.1 # limit on the actuations
x_init_val = [-0.01, 0.03, 30*(np.pi/180.), 0]
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
N_MPC = np.sum(Mm) # time horizon for the MPC controller
N_m = Mm.size
N = T*freq # total number of iterations
dt = 1.0/freq # sampling time
N_var = (N_xu)*N_MPC
h = 1./freq # time interval of each iteration
A = a**2 # area of the slider in meter square
f_max = miu_g*m*g # limit force in Newton
# Area integral of norm of the distance for a square:
int_square = lambda a: dblquad(lambda x,y: np.sqrt(x**2 + y**2), -a/2, a/2, -a/2, a/2)[0]
int_A = int_square(a)
m_max = miu_g*m*g*int_A/A # limit torque Newton meter
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
# z[1] - Sliding Left mode
# z[2] - Sliding Right mode
z = cs.SX.sym('x', N_i)
#  ------------------------------------------------------------------

## Build Motion Model
#  -------------------------------------------------------------------
R_pusher_func = my_dynamics.square_slider_quasi_static_ellipsoidal_limit_surface_R
#  -------------------------------------------------------------------
p_pusher_func = cs.Function('p_pusher_func', [x], [my_dynamics.square_slider_quasi_static_ellipsoidal_limit_surface_p(x, beta)], ['x'], ['p'])
#  -------------------------------------------------------------------
f_func = cs.Function('f_func', [x,u], [my_dynamics.square_slider_quasi_static_ellipsoidal_limit_surface_f(x,u, beta)],['x','u'],['xdot'])
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
fric_cone_c = cs.Function('fric_cone_c', [u_bar], [cs.vertcat(miu_p*u_bar[0]+u_bar[1], miu_p*u_bar[0]-u_bar[1])])
fric_cone_C = fric_cone_c.map(N-1)
#  -------------------------------------------------------------------

## Generate Nominal Trajectory
#  -------------------------------------------------------------------
# x0_nom, x1_nom = my_trajectories.generate_traj_line(0.5, 0.0, N)
# x0_nom, x1_nom = my_trajectories.generate_traj_line(0.5, 0.3, N)
# x0_nom, x1_nom = my_trajectories.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.25, N)
x0_nom, x1_nom = my_trajectories.generate_traj_eight(0.2, N)
#  -------------------------------------------------------------------
# stack state and derivative of state
X_nom_val, dX_nom_val = my_trajectories.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  ------------------------------------------------------------------
# control path variables
u_nom = cs.SX.sym('u_nom', N_u, N-1)
#  ------------------------------------------------------------------
# declare cost function
W_f = cs.diag(cs.SX([1.0,1.0,0.01,0.01]))
vel_error = dx - f_func(x, u)
cost_f = cs.Function('cost', [x, dx, u], [cs.dot(vel_error,cs.mtimes(W_f,vel_error))])
cost_F = cost_f.map(N-1)
#  -------------------------------------------------------------------
opt = my_opt.OptVars()
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
args = my_opt.OptArgs()
# initial condition for opt var
args.x0 = [0.0]*((N-1)*N_u)
# opt var boundaries
args.lbx = [-cs.inf, -cs.inf, 0.0]*(N-1)
args.ubx = [cs.inf, cs.inf, 0.0]*(N-1)
# arg for sticking constraint
args.lbg = [0.0]*((N-1)*2)
args.ubg = [cs.inf]*((N-1)*2)
#  -------------------------------------------------------------------
# Solve optimization problem
sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg)
u_sol = sol['x']
U_nom_val = np.array(cs.horzcat(u_sol[0::N_u],u_sol[1::N_u],u_sol[2::N_u]).T)
#  -------------------------------------------------------------------

## Compute argumens for the entire nominal trajectory
#  -------------------------------------------------------------------
ARGS_NOM = my_opt.OptArgs()
## ---- Initialize variables for optimization problem ---
ARGS_NOM.lbg = []
ARGS_NOM.ubg = []
ARGS_NOM.lbx = []
ARGS_NOM.ubx = []
ARGS_NOM.p = []
for i in range(N-1):
    ## ---- Dynamic constraints ----
    ARGS_NOM.lbg += [0]*N_x
    ARGS_NOM.ubg += [0]*N_x
    ## ---- State constraints ----
    ARGS_NOM.lbg += [-U_nom_val[2,i], -cs.inf]
    ARGS_NOM.ubg += [cs.inf, -U_nom_val[2,i]]
    ## ---- Control constraints ----
    ARGS_NOM.lbg += (-fric_cone_c(U_nom_val[:,i])).elements()
    ARGS_NOM.ubg += [cs.inf]*2
    ARGS_NOM.lbg += [-cs.inf]*2
    ARGS_NOM.ubg += (-fric_cone_c(U_nom_val[:,i])).elements()
    ## ---- Add States to optimization variables ---
    ARGS_NOM.lbx += [-cs.inf]*N_x
    ARGS_NOM.ubx += [cs.inf]*N_x
    ## ---- Add Actions to optimization variables ---
    ARGS_NOM.lbx += [-U_nom_val[0,i], -f_lim-U_nom_val[1,i], -cs.inf]
    ARGS_NOM.ubx += [f_lim-U_nom_val[0,i], f_lim-U_nom_val[1,i], cs.inf]
    ## ---- Set nominal trajectory as parameters ----
    ARGS_NOM.p += X_nom_val[:,i].tolist()
    ARGS_NOM.p += U_nom_val[:,i].tolist()
## ---- Add last States to optimization variables ---
ARGS_NOM.lbx += [-cs.inf]*N_x
ARGS_NOM.ubx += [cs.inf]*N_x
ARGS_NOM.p += X_nom_val[:,-1].tolist()
#  -------------------------------------------------------------------

## Define variables for optimization
#  -------------------------------------------------------------------
## ---- Input variables ---
X_bar = cs.SX.sym('x_bar', N_x, N_MPC)
U_bar = cs.SX.sym('u_bar', N_u, N_MPC-1)
## ---- Nominal state and action variables ----
X_nom = cs.SX.sym('x_nom', N_x, N_MPC)
U_nom = cs.SX.sym('u_nom', N_u, N_MPC-1)
## ---- Initial state and action variables ----
x_init = cs.SX.sym('x0', N_x)
u_init = cs.SX.sym('u0', N_u)
## ---- discrete variables ----
Zm = cs.SX.sym('z', N_i, N_m)
Z = cs.repmat(Zm[:, 0], 1, Mm[0])
for i in range(1, N_m):
    Z = cs.horzcat(Z, cs.repmat(Zm[:, i], 1, Mm[i]))
#  -------------------------------------------------------------------

## Set up QP Optimization Problem
#  -------------------------------------------------------------------
opt = my_opt.OptVars()
## ---- Set optimization objective ----------
Qcost = cs.diag(cs.SX([3.0,3.0,0.01,0])); QcostN = 200*Qcost
Rcost = cs.diag(cs.SX([1,1,0.2]))
wcost = cs.SX([0.0, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
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
opt.discrete = []
for i in range(N_MPC-1):
    opt.x += X_bar[:,i].elements()
    opt.discrete += [False]*N_x
    opt.x += U_bar[:,i].elements()
    opt.discrete += [False]*N_u
opt.x += X_bar[:,-1].elements()
opt.discrete += [False]*N_x
for i in range(N_m):
    opt.x += Zm[:,i].elements()
    opt.discrete += [True]*N_i
## ---- Set optimzation constraints ----
opt.g = []
opt.g += [X_bar[:,0]+X_nom[:,0]-x_init] ## Initial Conditions
for i in range(N_MPC-1):
    ## Dynamic constraints
    opt.g += dyn_err_f(X_nom[:,i], U_nom[:,i], X_bar[:,i], X_bar[:,i+1], U_bar[:,i]).elements()
    ## State constraints
    opt.g += [U_bar[2,i] + bigM*Z[2,i]]
    opt.g += [U_bar[2,i] - bigM*Z[1,i]]
    ## Control constraints
    opt.g += (fric_cone_c(U_bar[:,i]) + bigM*Z[0:2,i]).elements()
    opt.g += (fric_cone_c(U_bar[:,i]) - bigM*(1-Z[0:2,i])).elements()
for i in range(N_m):
    ## Integer summation
    opt.g += [cs.sum1(Zm[:,i])]
## ---- Set optimization parameters ----
opt.p = []
opt.p += x_init.elements()
opt.p += u_init.elements()
for i in range(N_MPC-1):
    opt.p += X_nom[:,i].elements()
    opt.p += U_nom[:,i].elements()
opt.p += X_nom[:,-1].elements()
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
Nidx = N-N_MPC
# Nidx = 10
X_plot = np.empty([N_x, Nidx+1])
U_plot = np.empty([N_u, Nidx])
X_plot[:,0] = x_init_val
X_future = np.empty([N_x, N_MPC, Nidx])
comp_time = np.empty(Nidx)
#  -------------------------------------------------------------------

z_val = cs.SX.zeros(3,1)
z_val[0] = 1
Zm0 = np.array(cs.DM(cs.repmat(z_val, 1, N_m)))
Zm_lbx = np.zeros(N_m*N_i)
Zm_ubx =  np.ones(N_m*N_i)
Zm_bg = np.ones(N_m)
## Set arguments and solve
#  -------------------------------------------------------------------
x0 = x_init_val
u0 = u_init_val
args = my_opt.OptArgs()
for idx in range(Nidx):
    # Indexing
    idx_x_i = idx*(N_xu)
    idx_x_f = (idx+N_MPC-1)*(N_xu)+N_x
    idx_g_i = idx*N_g
    idx_g_f = (idx+N_MPC-1)*N_g
    # warm start
    if idx==0:
        args.x0 = ARGS_NOM.p[idx_x_i:idx_x_f]
        args.x0 += Zm0.flatten('F').tolist()
    else:
        args.x0 = x_opt[6:-1].elements()
        args.x0 += ARGS_NOM.p[(idx_x_f-(N_xu)):idx_x_f]
        args.x0 += x_i.elements()
    # setting optimization bounderies from nominal traj
    args.lbx = ARGS_NOM.lbx[idx_x_i:idx_x_f]
    args.lbx += Zm_lbx.tolist()
    args.ubx = ARGS_NOM.ubx[idx_x_i:idx_x_f]
    args.ubx += Zm_ubx.tolist()
    ## setting parameters
    args.p = []
    args.p += x0
    args.p += u0
    args.p += ARGS_NOM.p[idx_x_i:idx_x_f]
    # initial state constraint
    args.lbg = [0]*N_x
    args.ubg = [0]*N_x
    # dynamics and friction constraints
    args.lbg += ARGS_NOM.lbg[idx_g_i:idx_g_f]
    args.lbg += Zm_bg.tolist()
    args.ubg += ARGS_NOM.ubg[idx_g_i:idx_g_f]
    args.ubg += Zm_bg.tolist()
    ## ---- Solve the optimization ----
    start_time = time.time()
    sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg, p=args.p)
    x_opt = sol['x'][0:(N_MPC*N_xu-N_u)]
    z_opt = sol['x'][(N_MPC*N_xu-N_u):-1]
    # print(z_opt[0:3])
    x_i = sol['x'][(N_MPC*N_xu-N_u):]
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
    #x0 = X_opt[:,1].elements()
    x0 = (x0 + f_func(x0, u0)*dt).elements()
    ## ---- Store values for plotting ----
    X_plot[:,idx+1] = x0
    U_plot[:,idx] = u0
    X_future[:,:,idx] = np.array(X_opt)
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(7,9))
#  -------------------------------------------------------------------
t_N_x = np.linspace(0, T, N)
t_N_u = np.linspace(0, T, N-1)
t_idx_x = t_N_x[0:Nidx+1]
t_idx_u = t_N_x[0:Nidx]
X_nom_val = np.array(X_nom_val)
#  -------------------------------------------------------------------
axs[0].plot(t_N_x, X_nom_val[0,:], 'b', label='x nom')
axs[0].plot(t_idx_x, X_plot[0,:], '--g', label='x opt')
axs[0].plot(t_N_x, X_nom_val[1,:], 'r', label='y nom')
axs[0].plot(t_idx_x, X_plot[1,:], '--y', label='y opt')
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels)
axs[0].set_ylabel('position [m]')
axs[0].set_title('Slider CoM')
axs[0].grid()
#  -------------------------------------------------------------------
axs[1].plot(t_N_x, X_nom_val[2,:]*(180/np.pi), 'b', label='slider nom')
axs[1].plot(t_idx_x, X_plot[2,:]*(180/np.pi), '--g', label='slider opt')
axs[1].plot(t_N_x, X_nom_val[3,:]*(180/np.pi), 'r', label='pusher nom')
axs[1].plot(t_idx_x, X_plot[3,:]*(180/np.pi), '--y', label='pusher opt')
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles, labels)
axs[1].set_ylabel('angles [degrees]')
axs[1].set_title('Angles of pusher and Slider')
axs[1].grid()
#  -------------------------------------------------------------------
axs[2].plot(t_N_u, U_nom_val[0,:], 'b', label='norm nom')
axs[2].plot(t_idx_u, U_plot[0,:], '--g', label='norm bar')
axs[2].plot(t_N_u, U_nom_val[1,:], 'r', label='tan nom')
axs[2].plot(t_idx_u, U_plot[1,:], '--y', label='tan bar')
handles, labels = axs[2].get_legend_handles_labels()
axs[2].legend(handles, labels)
axs[2].set_ylabel('vel [m/s]')
axs[2].set_title('Puhser control vel')
axs[2].grid()
#  -------------------------------------------------------------------
axs[3].plot(t_idx_u, comp_time)
axs[3].set_xlabel('time [s]')
axs[3].set_ylabel('time [s]')
axs[3].set_title('Computational time')
axs[3].grid()
#  -------------------------------------------------------------------
plt.show(block=False)
#  -------------------------------------------------------------------

# Animation
#  -------------------------------------------------------------------
if show_anim:
#  -------------------------------------------------------------------
    fig, ax = my_plots.plot_nominal_traj(x0_nom, x1_nom)
    # get slider and pusher patches
    slider, pusher, path_past, path_future = my_plots.get_patches_for_square_slider_and_cicle_pusher(
            ax, 
            p_pusher_func, 
            R_pusher_func, 
            X_plot,
            a, r_pusher)
    # call the animation
    ani = animation.FuncAnimation(fig,
            my_plots.animate_square_slider_and_circle_pusher,
            fargs=(slider, pusher, ax, p_pusher_func, R_pusher_func, X_plot, a, path_past, path_future, X_future),
            frames=Nidx,
            interval=dt*1000,
            blit=True,
            repeat=False)
    ## to save animation, uncomment the line below:
    ## ani.save('sliding_nominal_traj.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
