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
import sliding_pack
#  -------------------------------------------------------------------

## Set Problem constants
#  -------------------------------------------------------------------
N_x = 4 # number of state variables
N_u = 3 # number of actions variables
N_xu = N_x + N_u
a = 0.09 # side dimension of the square slider in meters
T = 2 # time of the simulation is seconds
freq = 25 # number of increments per second
r_pusher = 0.01 # radius of the cylindrical pusher in meter
miu_p = 0.2 # coefficient of friction between pusher and slider
W_goal = cs.diag(cs.SX([1.0,1.0,0.01,0.0]))
W_dx = 0.00001*cs.diag(cs.SX([1.0,1.0,0.01,0.0]))
f_lim = 0.3 # limit on the actuations
x_init_val = [-0.01, 0.03, 30*(np.pi/180.), 0]
# x_end_val = [0.0, 0.5, 180*(np.pi/180.), 0]
x_end_val = [0.2, 0.5, 180*(np.pi/180.), 0]
u_init_val = [0.0, 0.0, 0.0, 0.0]
show_anim = True
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N = int(T*freq) # total number of iterations
dt = 1.0/freq
#  -------------------------------------------------------------------

## Define state and control vectors
#  -------------------------------------------------------------------
# control path variables
u_nom = cs.SX.sym('u_nom', N_u, N-1)
x0 = cs.SX.sym('x0', N_x)
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
# u[0] - normal vel in the local frame
# u[1] - tangential vel in the local frame
# u[2] - relative sliding velocity between pusher and slider
u = cs.SX.sym('u', N_u)
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

## Define constraint functions ----
#  -------------------------------------------------------------------
x_next = cs.SX.sym('x_next', N_x)
## ---- Define Dynamic constraints ----
dyn_err_f = cs.Function('dyn_err_f', [x, u, x_next], 
        [x_next-x-dt*f_func(x,u)])
## ---- Define Control constraints ----
fric_cone_c = cs.Function('fric_cone_c', [u], [cs.vertcat(miu_p*u[0]+u[1], miu_p*u[0]-u[1])])
#  -------------------------------------------------------------------

## initialize variables for optimization problem
#  -------------------------------------------------------------------
## ---- Input variables ---
X = cs.SX.sym('X', N_x, N)
U = cs.SX.sym('U', N_u, N-1)
#  -------------------------------------------------------------------
## ---- Initialize optimization and argument variables ---
opt = sliding_pack.opt.OptVars()
args = sliding_pack.opt.OptArgs()
## ---- Define cost function ----
goal_error = x - x_end_val
cost_goal = cs.Function('cost_goal', [x], [cs.dot(goal_error,cs.mtimes(W_goal,goal_error))])
cost_goal_traj = cost_goal.map(N)
dx = f_func(x,u)
cost_dx = cs.Function('cost_dx', [x,u], [cs.dot(dx,cs.mtimes(W_dx,dx))])
cost_dx_traj = cost_dx.map(N-1)
opt.f = cost_goal(X[:,-1])
# opt.f += cs.sum2(cost_dx_traj(X[:,0:-1], U))
# sys.exit()
# opt.f = cs.sum2(cost_goal_traj(X))
## ---- initial state constraint ----
opt.g = (X[:,0]-x_init_val).elements()
args.lbg = [0.0]*N_x
args.ubg = [0.0]*N_x
for i in range(N-1):
    ## ---- dynamics constraint ----
    opt.g += dyn_err_f(X[:,i], U[:,i], X[:,i+1]).elements()
    args.lbg += [0.0]*N_x
    args.ubg += [0.0]*N_x
for i in range(N-1):
    ## ---- friction cone constraint ----
    opt.g += fric_cone_c(U[:,i]).elements()
    args.lbg += [0.0]*2
    args.ubg += [cs.inf]*2
## ---- Define optimization variable ----
x_nom = np.linspace(x_init_val, x_end_val, N).transpose()
opt.x = []
args.x0 = []
args.lbx = []
args.ubx = []
for i in range(N-1):
    ## ---- Add States to optimization variables ---
    opt.x    += [X[:,i]]
    args.lbx += [-cs.inf]*N_x
    args.ubx += [cs.inf]*N_x
    args.x0  += [x_nom[:,i]]
    ## ---- Add Actions to optimization variables ---
    # actions: normal vel, tangential vel, relative sliding vel
    opt.x    += [U[:,i]]
    args.lbx += [0.0,  -f_lim, 0.0]
    args.ubx += [f_lim, f_lim, 0.0]
    args.x0  += [0.0,     0.0, 0.0]
## ---- Add last States to optimization variables ---
opt.x += [X[:,-1]]
args.lbx += [-cs.inf]*N_x
args.ubx += [cs.inf]*N_x
args.x0 += [x_nom[:,-1]]
#  -------------------------------------------------------------------

## Generating solver
#  -------------------------------------------------------------------
prob = {'f': opt.f, 'x': cs.vertcat(*opt.x), 'g': cs.vertcat(*opt.g)}
solver = cs.nlpsol('solver', 'ipopt', prob)
# solver = cs.nlpsol('solver', 'snopt', prob)
#  -------------------------------------------------------------------

## Solve optimization problem
#  -------------------------------------------------------------------
sol = solver(x0=cs.vertcat(*args.x0), lbx=cs.vertcat(*args.lbx), ubx=cs.vertcat(*args.ubx), lbg=cs.vertcat(*args.lbg), ubg=cs.vertcat(*args.ubg))
x_sol = sol['x']
x_opt = cs.horzcat(x_sol[0::N_xu],x_sol[1::N_xu],x_sol[2::N_xu],x_sol[3::N_xu]).T
u_opt = cs.horzcat(x_sol[4::N_xu],x_sol[5::N_xu],x_sol[6::N_xu]).T
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12,8))
ts = np.linspace(0, T, N)
tds = np.linspace(0, T, N-1)
#  -------------------------------------------------------------------
axs[0,0].plot(ts, x_nom[0,:].T, color='b', label='nom')
axs[0,0].plot(ts, x_opt[0,:].T, color='g', linestyle='--', label='opt')
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles, labels)
axs[0,0].set_ylabel('x0')
axs[0,0].grid()
#  -------------------------------------------------------------------
axs[1,0].plot(ts, x_nom[1,:].T, color='b', label='nom')
axs[1,0].plot(ts, x_opt[1,:].T, color='g', linestyle='--', label='opt')
handles, labels = axs[1,0].get_legend_handles_labels()
axs[1,0].legend(handles, labels)
axs[1,0].set_ylabel('x1')
axs[1,0].grid()
#  -------------------------------------------------------------------
axs[2,0].plot(ts, x_nom[2,:].T*(180/np.pi), color='b', label='nom')
axs[2,0].plot(ts, x_opt[2,:].T*(180/np.pi), color='g', linestyle='--', label='opt')
handles, labels = axs[2,0].get_legend_handles_labels()
axs[2,0].legend(handles, labels)
axs[2,0].set_ylabel('x2')
axs[2,0].grid()
#  -------------------------------------------------------------------
axs[0,1].plot(ts, x_nom[3,:].T*(180/np.pi), color='b', label='nom')
axs[0,1].plot(ts, x_opt[3,:].T*(180/np.pi), color='g', linestyle='--', label='opt')
handles, labels = axs[0,1].get_legend_handles_labels()
axs[0,1].legend(handles, labels)
axs[0,1].set_ylabel('x3')
axs[0,1].grid()
#  -------------------------------------------------------------------
axs[1,1].plot(tds, u_opt[0,:].T, color='b', label='u0')
axs[1,1].plot(tds, u_opt[1,:].T, color='r', label='u1')
axs[1,1].plot(tds, u_opt[2,:].T, color='g', label='u2')
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].legend(handles, labels)
axs[1,1].set_xlabel('time [s]')
axs[1,1].set_ylabel('u opt')
axs[1,1].grid()
#  -------------------------------------------------------------------

# Animation of Nominal Trajectory
#  -------------------------------------------------------------------
if show_anim:
#  -------------------------------------------------------------------
    x_anim = np.array(x_opt)
    fig, ax = my_plots.plot_nominal_traj(x_anim[0,:].T, x_anim[1,:].T)
    # get slider and pusher patches
    slider, pusher, _, _ = my_plots.get_patches_for_square_slider_and_cicle_pusher(
            ax, 
            p_pusher_func, 
            R_pusher_func, 
            x_anim,
            a, r_pusher)
    # call the animation
    ani = animation.FuncAnimation(
            fig,
            my_plots.animate_square_slider_and_circle_pusher,
            fargs=(slider, pusher, ax, p_pusher_func, R_pusher_func, x_anim, a),
            frames=N,
            interval=dt*1000,
            blit=True,
            repeat=False)
    ## to save animation, uncomment the line below:
    # ani.save('sliding_nominal_traj.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
