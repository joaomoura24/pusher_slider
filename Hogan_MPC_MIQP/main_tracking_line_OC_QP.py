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
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
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
a = 0.09 # side dimension of the square slider in meters
miu_p = 0.3 # coeficient of friction between pusher and slider
T = 8 # time of the simulation is seconds
freq = 50 # numer of increments per second
r_pusher = 0.01 # radious of the cilindrical pusher in meter
x_init = [-0.01, 0.03, 30*(np.pi/180.), 0] # initial state
show_anim = True
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N = T*freq # total number of iterations
dt = 1.0/freq # time interval of each iteration
#  -------------------------------------------------------------------

## Define state and control vectors
#  -------------------------------------------------------------------
# x - state vector
# x[0] - x slider CoM position in the global frame
# x[1] - y slider CoM position in the global frame
# x[2] - slider orientation in the global frame
# x[3] - angle of pusher relative to slider
x = cs.SX.sym('x', 4)
# dx - state vector derivative
dx = cs.SX.sym('dx', 4)
# u - control vector
# u[0] - normal force in the local frame
# u[1] - tangential force in the local frame
# u[2] - relative sliding velocity between pusher and slider
u = cs.SX.sym('u', 3)
# b - dynamic parameters
# b[0] - slider lenght [m]
# b[1] - radious of the pusher [m]
beta = [a, r_pusher]
#  -------------------------------------------------------------------

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
fric_cone_c = cs.Function('fric_cone_c', [u], [cs.vertcat(miu_p*u[0]+u[1], miu_p*u[0]-u[1])])
fric_cone_C = fric_cone_c.map(N-1)
#  -------------------------------------------------------------------

## Generate Nominal Trajectory
#  -------------------------------------------------------------------
# x0_nom, x1_nom = my_trajectories.generate_traj_line(0.5, 0.0, N, 0)
# x0_nom, x1_nom = my_trajectories.generate_traj_line(0.5, 0.3, N, 0)
# x0_nom, x1_nom = my_trajectories.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.15, N, 0)
x0_nom, x1_nom = my_trajectories.generate_traj_eight(0.2, N, 0)
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
args.lbx = [0.0, -cs.inf, 0.0]*(N-1)
args.ubx = [cs.inf, cs.inf, 0.0]*(N-1)
# arg for sticking constraint
args.lbg = [0.0]*((N-1)*2)
args.ubg = [cs.inf]*((N-1)*2)
#  -------------------------------------------------------------------
# Solve optimization problem
sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg)
u_sol = sol['x']
U_nom_val = cs.horzcat(u_sol[0::N_u],u_sol[1::N_u],u_sol[2::N_u]).T
#  -------------------------------------------------------------------

## Set up QP Optimization Problem
#  -------------------------------------------------------------------
X_nom = cs.SX.sym('X_nom', N_x, N)
U_nom = cs.SX.sym('U_nom', N_u, N-1)
X_bar = cs.SX.sym('X_bar', N_x, N)
U_bar = cs.SX.sym('U_bar', N_u, N-1)
X = X_nom + X_bar
U = U_nom + U_bar
## ---- Define Optimization objective ---
Qcost = cs.diag(cs.SX([3.0,3.0,0.01,0]))
Rcost = cs.diag(cs.SX([1,1,0.0]))
cost_f = cs.Function('cost', [x, u], [cs.dot(x,cs.mtimes(Qcost,x)) + cs.dot(u,cs.mtimes(Rcost,u))])
cost_F = cost_f.map(N-1)
## ---- Initialize optimization and argument variables ---
opt = my_opt.OptVars()
args = my_opt.OptArgs()
## ---- passing parameters ----
opt.p = []
opt.p += x.elements()
opt.p += X_nom.elements()
opt.p += U_nom.elements()
args.p = []
args.p += x_init
args.p += X_nom_val.elements()
args.p += U_nom_val.elements()
## ---- cost function ----
opt.f = cs.sum2(cost_F(X_bar[:,0:-1], U_bar)) + cost_f(X_bar[:,-1], cs.SX(N_u, 1)) 
## ---- initial state constraint ----
opt.g = (X[:,0]-x_init).elements()
args.lbg = [0.0]*N_x
args.ubg = [0.0]*N_x
for i in range(N-1):
    ## ---- dynamics constraint ----
    opt.g += dyn_err_f(X_nom[:,i], U_nom[:,i], X_bar[:,i], X_bar[:,i+1], U_bar[:,i]).elements()
    args.lbg += [0.0]*N_x
    args.ubg += [0.0]*N_x
for i in range(N-1):
    ## ---- friction cone constraint ----
    opt.g += fric_cone_c(U[:,i]).elements()
    args.lbg += [0.0]*2
    args.ubg += [cs.inf]*2
#-----------------------
opt.x = []
args.x0 = []
args.lbx = []
args.ubx = []
# U_nom_val = np.array(U_nom_val)
for i in range(N-1):
    ## ---- Add States to optimization variables ---
    opt.x    += [X_bar[:,i]]
    args.lbx += [-cs.inf]*N_x
    args.ubx += [cs.inf]*N_x
    args.x0  += [X_nom_val[:,i]]
    ## ---- Add Actions to optimization variables ---
    # actions: normal vel, tangential vel, relative sliding vel
    opt.x    += [U_bar[:,i]]
    args.lbx += [-U_nom_val[0,i], -cs.inf, U_nom_val[2,i]]
    args.ubx += [cs.inf,           cs.inf, U_nom_val[2,i]]
    args.x0  += [0.0,                 0.0, U_nom_val[2,i]]
## ---- Add last States to optimization variables ---
opt.x += [X_bar[:,-1]]
args.lbx += [-cs.inf]*N_x
args.ubx += [cs.inf]*N_x
args.x0 += [X_nom_val[:,-1]]
## ---- Create solver ----
prob = {'f': opt.f, 'x': cs.vertcat(*opt.x), 'g': cs.vertcat(*opt.g), 'p': cs.vertcat(*opt.p)}
solver = cs.nlpsol('solver', 'ipopt', prob)
# solver = cs.nlpsol('solver', 'snopt', prob)
# solver = cs.qpsol('S', 'qpoases', prob, {'sparse':True})
# solver = cs.qpsol('solver', 'gurobi', prob)
## ---- Solve optimization problem ----
sol = solver(x0=cs.vertcat(*args.x0), lbx=cs.vertcat(*args.lbx), ubx=cs.vertcat(*args.ubx), lbg=cs.vertcat(*args.lbg), ubg=cs.vertcat(*args.ubg), p=args.p)
w_opt = sol['x']
my_plots.plot_sparsity(cs.horzcat(*opt.g), cs.vertcat(*opt.x), w_opt)
## ---- Compute actual trajectory and controls ----
X_bar_opt = np.array(cs.horzcat(w_opt[0::7],w_opt[1::7],w_opt[2::7],w_opt[3::7]).T)
U_bar_opt = np.array(cs.horzcat(w_opt[4::7],w_opt[5::7],w_opt[6::7]).T)
X_opt = X_bar_opt + X_nom_val
U_opt = U_bar_opt + U_nom_val
cost_opt = cost_F(X_bar_opt[:,0:-1], U_bar_opt).T
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(7,9))
#  -------------------------------------------------------------------
ts = np.linspace(0, T, N)
axs[0].plot(ts, X_nom_val[0,:].T, 'b', label='x nom')
axs[0].plot(ts, X_opt[0,:].T, '--g', label='x opt')
axs[0].plot(ts, X_nom_val[1,:].T, 'r', label='y nom')
axs[0].plot(ts, X_opt[1,:].T, '--y', label='y opt')
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels)
axs[0].set_ylabel('position [m]')
axs[0].set_title('Slider CoM')
axs[0].grid()
#  -------------------------------------------------------------------
axs[1].plot(ts, X_nom_val[2,:].T*(180/np.pi), 'b', label='slider nom')
axs[1].plot(ts, X_opt[2,:].T*(180/np.pi), '--g', label='slider opt')
axs[1].plot(ts, X_nom_val[3,:].T*(180/np.pi), 'r', label='pusher nom')
axs[1].plot(ts, X_opt[3,:].T*(180/np.pi), '--y', label='pusher opt')
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles, labels)
axs[1].set_ylabel('angles [degrees]')
axs[1].set_title('Angles of pusher and Slider')
axs[1].grid()
#  -------------------------------------------------------------------
ts = np.linspace(0, T, N-1)
axs[2].plot(ts, U_nom_val[0,:].T, 'b', label='norm nom')
axs[2].plot(ts, U_bar_opt[0,:], '--g', label='norm bar')
axs[2].plot(ts, U_nom_val[1,:].T, 'r', label='tan nom')
axs[2].plot(ts, U_bar_opt[1,:], '--y', label='tan bar')
handles, labels = axs[2].get_legend_handles_labels()
axs[2].legend(handles, labels)
axs[2].set_ylabel('vel [m/s]')
axs[2].set_title('Puhser control vel')
axs[2].grid()
#  -------------------------------------------------------------------
axs[3].plot(ts, np.array(cost_opt), color='b', label='cost')
axs[3].set_xlabel('time [s]')
axs[3].set_ylabel('cost ')
axs[3].set_title('cost along traj.')
axs[3].grid()
#  -------------------------------------------------------------------

# Animation of Nominal Trajectory
#  -------------------------------------------------------------------
if show_anim:
#  -------------------------------------------------------------------
    x_anim = np.array(X_opt)
    fig, ax = my_plots.plot_nominal_traj(x0_nom, x1_nom)
    # get slider and pusher patches
    slider, pusher, path, _ = my_plots.get_patches_for_square_slider_and_cicle_pusher(
            ax, 
            p_pusher_func, 
            R_pusher_func, 
            x_anim,
            a, r_pusher)
    # call the animation
    ani = animation.FuncAnimation( fig,
            my_plots.animate_square_slider_and_circle_pusher,
            fargs=(slider, pusher, ax, p_pusher_func, R_pusher_func, x_anim, a, path),
            frames=N,
            interval=dt*1000,
            blit=True,
            repeat=False)
    ## to save animation, uncomment the line below:
    ## ani.save('sliding_nominal_traj.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
