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
import sliding_pack
#  -------------------------------------------------------------------

## Set Problem constants
#  -------------------------------------------------------------------
N_x = 4 # number of state variables
N_u = 3 # number of actions variables
g = 9.81 # gravity acceleration constant in meter per second square
a = 0.09 # side dimension of the square slider in meters
m = 0.827 # mass of the slider in kilo grams
miu_g = 0.35 # coeficient of friction between slider and table
miu_p = 0.3 # coeficient of friction between pusher and slider
T = 10 # time of the simulation is seconds
freq = 50 # numer of increments per second
r_pusher = 0.01 # radious of the cilindrical pusher in meter
N_MPC = 200 # time horizon for the MPC controller
x_init_val = [-0.01, 0.03, 30*(np.pi/180.), 0]
u_init_val = [0.0, 0.0, 0.0]
show_anim = True
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N_xu = N_x + N_u # number of optimization variables
N = T*freq # total number of iterations
dt = 1.0/freq # time interval of each iteration
T_MPC = N_MPC*dt
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
#  -------------------------------------------------------------------

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
fric_cone_c = cs.Function('fric_cone_c', [u_bar], [cs.vertcat(miu_p*u_bar[0]+u_bar[1], miu_p*u_bar[0]-u_bar[1])])
fric_cone_C = fric_cone_c.map(N-1)
#  -------------------------------------------------------------------

## Generate Nominal Trajectory
#  -------------------------------------------------------------------
x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.5, 0.0, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.5, 0.3, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.25, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.5, N, 0)
#  -------------------------------------------------------------------
# stack state and derivative of state
X_nom_val, dX_nom_val = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
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
ARGS_NOM = sliding_pack.opt.OptArgs()
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
    ## ---- Control constraints ----
    ARGS_NOM.lbg += (-fric_cone_c(U_nom_val[:,i])).elements()
    ARGS_NOM.ubg += [cs.inf]*2
    ## ---- Add States to optimization variables ---
    ARGS_NOM.lbx += [-cs.inf]*N_x
    ARGS_NOM.ubx += [cs.inf]*N_x
    ## ---- Add Actions to optimization variables ---
    # [normal vel, tangential vel, relative sliding vel]
    ARGS_NOM.lbx += [-U_nom_val[0,i], -cs.inf, U_nom_val[2,i]]
    ARGS_NOM.ubx += [cs.inf, cs.inf, U_nom_val[2,i]]
    ## ---- Set nominal trajectory as parameters ----
    ARGS_NOM.p += X_nom_val[:,i].elements()
    ARGS_NOM.p += U_nom_val[:,i].tolist()
## ---- Add last States to optimization variables ---
ARGS_NOM.lbx += [-cs.inf]*N_x
ARGS_NOM.ubx += [cs.inf]*N_x
ARGS_NOM.p += X_nom_val[:,-1].elements()
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
#  -------------------------------------------------------------------

## Set up QP Optimization Problem
#  -------------------------------------------------------------------
## ---- Define Optimization objective ---
Qcost = cs.diag(cs.SX([3.0,3.0,0.01,0]))
Rcost = cs.diag(cs.SX([1,1,0.0]))
cost_f = cs.Function('cost', [x, u], [cs.dot(x,cs.mtimes(Qcost,x)) + cs.dot(u,cs.mtimes(Rcost,u))])
cost_F = cost_f.map(N_MPC-1)
## ---- Initialize optimization and argument variables ---
opt = sliding_pack.opt.OptVars()
## ---- cost function ----
opt.f = cs.sum2(cost_F(X_bar[:,0:-1], U_bar)) + cost_f(X_bar[:,-1], cs.SX(N_u, 1)) 
## ---- Set optimization variables ----
opt.x = []
for i in range(N_MPC-1):
    opt.x += X_bar[:,i].elements()
    opt.x += U_bar[:,i].elements()
opt.x += X_bar[:,-1].elements()
## ---- Set optimzation constraints ----
opt.g = (X_bar[:,0]+X_nom[:,0]-x_init).elements() ## Initial Conditions
for i in range(N_MPC-1):
    ## ---- dynamics constraint ----
    opt.g += dyn_err_f(X_nom[:,i], U_nom[:,i], X_bar[:,i], X_bar[:,i+1], U_bar[:,i]).elements()
    ## ---- friction cone constraint ----
    opt.g += fric_cone_c(U_bar[:,i]).elements()
## ---- Set optimization parameters ----
opt.p = x_init.elements()
opt.p += u_init.elements()
for i in range(N_MPC-1):
    opt.p += X_nom[:,i].elements()
    opt.p += U_nom[:,i].elements()
opt.p += X_nom[:,-1].elements()
## ---- Create solver ----
prob = {'f': opt.f, 'x': cs.vertcat(*opt.x), 'g': cs.vertcat(*opt.g), 'p': cs.vertcat(*opt.p)}
solver = cs.nlpsol('solver', 'ipopt', prob)
#solver = cs.nlpsol('solver', 'snopt', prob)
#solver = cs.qpsol('S', 'qpoases', prob, {'sparse':True})
#solver = cs.qpsol('solver', 'gurobi', prob)
#  -------------------------------------------------------------------

## Set arguments and solve
#  -------------------------------------------------------------------
args = sliding_pack.opt.OptArgs()
# Indexing
idx = 0
idx_x_i = idx*N_xu
idx_x_f = (idx+N_MPC-1)*N_xu+N_x
# warm start
args.x0 = ARGS_NOM.p[idx_x_i:idx_x_f]
# optimization bounderies
args.lbx = ARGS_NOM.lbx[idx_x_i:idx_x_f]
args.ubx = ARGS_NOM.ubx[idx_x_i:idx_x_f]
## parameters
args.p = x_init_val
args.p += u_init_val
args.p += ARGS_NOM.p[idx_x_i:idx_x_f]
## constraints bounderies
idx_g_i = idx*6
idx_g_f = (idx+N_MPC-1)*6
# initial state constraint
args.lbg = [0]*N_x
args.ubg = [0]*N_x
# dynamics and friction constraints
args.lbg += ARGS_NOM.lbg[idx_g_i:idx_g_f]
args.ubg += ARGS_NOM.ubg[idx_g_i:idx_g_f]
## ---- Solve the optimization ----
sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg, p=args.p)
x_opt = sol['x']
sliding_pack.plots.plot_sparsity(cs.vertcat(*opt.g), cs.vertcat(*opt.x), x_opt)
## ---- Compute actual trajectory and controls ----
X_bar_opt = np.array(cs.horzcat(x_opt[0::N_xu],x_opt[1::N_xu],x_opt[2::N_xu],x_opt[3::N_xu]).T)
U_bar_opt = np.array(cs.horzcat(x_opt[4::N_xu],x_opt[5::N_xu],x_opt[6::N_xu]).T)
X_opt = X_bar_opt + X_nom_val[:,idx:(idx+N_MPC)]
U_opt = U_bar_opt + U_nom_val[:,idx:(idx+N_MPC-1)]
cost_opt = cost_F(X_opt[:,0:-1], U_opt).T
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(7,9))
#  -------------------------------------------------------------------
t_N_x = np.linspace(0, T, N)
t_N_u = np.linspace(0, T, N-1)
t_mpc_x = t_N_x[idx:(idx+N_MPC)]
t_mpc_u = t_N_x[idx:(idx+N_MPC-1)]
#  -------------------------------------------------------------------
axs[0].plot(t_N_x, X_nom_val[0,:].T, 'b', label='x nom')
axs[0].plot(t_mpc_x, X_opt[0,:].T, '--g', label='x opt')
axs[0].plot(t_N_x, X_nom_val[1,:].T, 'r', label='y nom')
axs[0].plot(t_mpc_x, X_opt[1,:].T, '--y', label='y opt')
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels)
axs[0].set_ylabel('position [m]')
axs[0].set_title('Slider CoM')
axs[0].grid()
#  -------------------------------------------------------------------
axs[1].plot(t_N_x, X_nom_val[2,:].T*(180/np.pi), 'b', label='slider nom')
axs[1].plot(t_mpc_x, X_opt[2,:].T*(180/np.pi), '--g', label='slider opt')
axs[1].plot(t_N_x, X_nom_val[3,:].T*(180/np.pi), 'r', label='pusher nom')
axs[1].plot(t_mpc_x, X_opt[3,:].T*(180/np.pi), '--y', label='pusher opt')
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles, labels)
axs[1].set_ylabel('angles [degrees]')
axs[1].set_title('Angles of pusher and Slider')
axs[1].grid()
#  -------------------------------------------------------------------
axs[2].plot(t_N_u, U_nom_val[0,:], 'b', label='norm nom')
axs[2].plot(t_mpc_u, U_opt[0,:], '--g', label='norm bar')
axs[2].plot(t_N_u, U_nom_val[1,:], 'r', label='tan nom')
axs[2].plot(t_mpc_u, U_opt[1,:], '--y', label='tan bar')
handles, labels = axs[2].get_legend_handles_labels()
axs[2].legend(handles, labels)
axs[2].set_ylabel('vel [m/s]')
axs[2].set_title('Puhser control vel')
axs[2].grid()
#  -------------------------------------------------------------------
axs[3].plot(t_mpc_u, np.array(cost_opt), color='b', label='cost')
axs[3].set_xlabel('time [s]')
axs[3].set_ylabel('cost ')
axs[3].set_title('cost along traj.')
axs[3].grid()
#  -------------------------------------------------------------------

# Animation
#  -------------------------------------------------------------------
if show_anim:
#  -------------------------------------------------------------------
    x_anim = np.array(X_opt)
    fig, ax = sliding_pack.plots.plot_nominal_traj(x0_nom, x1_nom)
    # get slider and pusher patches
    slider, pusher, path, _ = sliding_pack.plots.get_patches_for_square_slider_and_cicle_pusher(
            ax, 
            p_pusher_func, 
            R_pusher_func, 
            x_anim,
            a, r_pusher)
    # call the animation
    ani = animation.FuncAnimation(fig,
            sliding_pack.plots.animate_square_slider_and_circle_pusher,
            fargs=(slider, pusher, ax, p_pusher_func, R_pusher_func, x_anim, a, path),
            frames=N_MPC,
            interval=dt*1000,
            blit=True,
            repeat=False)
    ## to save animation, uncomment the line below:
    ## ani.save('sliding_nominal_traj.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
