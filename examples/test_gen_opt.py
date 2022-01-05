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
a = 0.09 # side dimension of the square slider in meters
T = 12 # time of the simulation is seconds
freq = 50 # number of increments per second
r_pusher = 0.01 # radius of the cylindrical pusher in meter
miu_p = 0.3 # coefficient of friction between pusher and slider
W_f = cs.diag(cs.SX([1.0,1.0,0.01,0.0]))
show_anim = True
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N = T*freq # total number of iterations
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
R_pusher_func = sliding_pack.dyn.square_slider_quasi_static_ellipsoidal_limit_surface_R
#  -------------------------------------------------------------------
p_pusher_func = cs.Function('p_pusher_func', [x], [sliding_pack.dyn.square_slider_quasi_static_ellipsoidal_limit_surface_p(x, beta)], ['x'], ['p'])
#  -------------------------------------------------------------------
f_func = cs.Function('f_func', [x,u], [sliding_pack.dyn.square_slider_quasi_static_ellipsoidal_limit_surface_f(x,u, beta)],['x','u'],['xdot'])
int_f_func = cs.Function('int_f_func', [x,u], [x+dt*f_func(x,u)])
#  -------------------------------------------------------------------
# compute time derivatives for all the input commands
F_func = f_func.map(N-1)
#  -------------------------------------------------------------------
# compute roll out function
# F_rollout = f_func.mapaccum(N-1)
F_rollout = int_f_func.mapaccum(N-1)
#  -------------------------------------------------------------------

## define control constraints ----
#  -------------------------------------------------------------------
fric_cone_c = cs.Function('fric_cone_c', [u], [cs.vertcat(miu_p*u[0]+u[1], miu_p*u[0]-u[1])])
fric_cone_c = fric_cone_c.map(N-1)
#  -------------------------------------------------------------------

## generate nominal trajectory
#  -------------------------------------------------------------------
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.25, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.5, 0.3, N, 0)
# x0_nom, x1_nom = sliding_pack.traj.generate_traj_line(0.5, 0.5, N, 0)
x0_nom, x1_nom = sliding_pack.traj.generate_traj_eight(0.2, N, 0)
#  ------------------------------------------------------------------
# stack state and derivative of state
x_nom, dx_nom = sliding_pack.traj.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  -------------------------------------------------------------------

## initialize variables for optimization problem
#  -------------------------------------------------------------------
# declare cost function
vel_error = dx - f_func(x, u)
cost_f = cs.Function('cost', [x, dx, u], [cs.dot(vel_error,cs.mtimes(W_f,vel_error))])
cost_F = cost_f.map(N-1)
#  -------------------------------------------------------------------
opt = sliding_pack.opt.OptVars()
# define cost function
opt.f = cs.sum2(cost_F(x_nom[:,0:-1], dx_nom, u_nom))
# define optimization variables
opt.x = cs.vertcat(*u_nom.elements())
# define Sticking constraint
# opt.g = cs.horzcat(*[
#     miu_p*u_nom[0,:]-u_nom[1,:], 
#     miu_p*u_nom[0,:]+u_nom[1,:]
# ])
# fric_cone_c = cs.Function('fric_cone_c', [u], [cs.vertcat(miu_p*u[0]+u[1], miu_p*u[0]-u[1])])
# opt.g = 
opt.g = cs.horzcat(*fric_cone_c(u_nom).elements())
#  -------------------------------------------------------------------

## Generating solver
#  -------------------------------------------------------------------
prob = {'f': opt.f, 'x': opt.x, 'g':opt.g}
solver = cs.nlpsol('solver', 'ipopt', prob)
# solver = cs.nlpsol('solver', 'snopt', prob)
#  -------------------------------------------------------------------

## Instanciating optimizer arguments
#  -------------------------------------------------------------------
args = sliding_pack.opt.OptArgs()
# initial condition for opt var
args.x0 = [0.0]*((N-1)*3)
# opt var boundaries
args.lbx = [-cs.inf, -cs.inf, 0.0]*(N-1)
args.ubx = [cs.inf, cs.inf, 0.0]*(N-1)
# arg for sticking constraint
args.lbg = [0.0]*((N-1)*2)
args.ubg = [cs.inf]*((N-1)*2)
#  -------------------------------------------------------------------

## Solve optimization problem
#  -------------------------------------------------------------------
sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg)
u_sol = sol['x']
u_opt = np.array(cs.horzcat(u_sol[0::N_u],u_sol[1::N_u],u_sol[2::N_u]).T)
cost_opt = cost_F(x_nom[:,0:-1], dx_nom, u_opt).T
total_cost_opt = sol['f']
#  -------------------------------------------------------------------
dx_opt = F_func(x_nom[:,0:-1], u_opt)
x_opt = cs.horzcat(x_nom[:,0],x_nom[:,0]+cs.cumsum(dx_opt*dt,2))
x_rollout = cs.horzcat(x_nom[:,0],F_rollout(x_nom[:,0], u_opt))
dx_rollout = cs.diff(x_rollout,1,1)/dt

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(5, 2, sharex=True, figsize=(12,8))
ts = np.linspace(0, T, N)
tds = np.linspace(0, T, N-1)
#  -------------------------------------------------------------------
axs[0,0].plot(ts, x_nom[0,:].T, color='b', label='nom')
axs[0,0].plot(ts, x_opt[0,:].T, color='g', linestyle='--', label='opt')
axs[0,0].plot(ts, x_rollout[0,:].T, color='r', linestyle=':', label='rollout')
handles, labels = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles, labels)
axs[0,0].set_ylabel('x0')
axs[0,0].grid()
#  -------------------------------------------------------------------
axs[1,0].plot(ts, x_nom[1,:].T, color='b', label='nom')
axs[1,0].plot(ts, x_opt[1,:].T, color='g', linestyle='--', label='opt')
axs[1,0].plot(ts, x_rollout[1,:].T, color='r', linestyle=':', label='rollout')
handles, labels = axs[1,0].get_legend_handles_labels()
axs[1,0].legend(handles, labels)
axs[1,0].set_ylabel('x1')
axs[1,0].grid()
#  -------------------------------------------------------------------
axs[2,0].plot(ts, x_nom[2,:].T*(180/np.pi), color='b', label='nom')
axs[2,0].plot(ts, x_opt[2,:].T*(180/np.pi), color='g', linestyle='--', label='opt')
axs[2,0].plot(ts, x_rollout[2,:].T*(180/np.pi), color='r', linestyle=':', label='rollout')
handles, labels = axs[2,0].get_legend_handles_labels()
axs[2,0].legend(handles, labels)
axs[2,0].set_ylabel('x2')
axs[2,0].grid()
#  -------------------------------------------------------------------
axs[3,0].plot(ts, x_nom[3,:].T*(180/np.pi), color='b', label='nom')
axs[3,0].plot(ts, x_opt[3,:].T*(180/np.pi), color='g', linestyle='--', label='opt')
axs[3,0].plot(ts, x_rollout[3,:].T*(180/np.pi), color='r', linestyle=':', label='rollout')
handles, labels = axs[3,0].get_legend_handles_labels()
axs[3,0].legend(handles, labels)
axs[3,0].set_ylabel('x3')
axs[3,0].grid()
#  -------------------------------------------------------------------
axs[4,0].plot(tds, u_opt[0,:], color='b', label='u0')
axs[4,0].plot(tds, u_opt[1,:], color='r', label='u1')
axs[4,0].plot(tds, u_opt[2,:], color='g', label='u2')
handles, labels = axs[4,0].get_legend_handles_labels()
axs[4,0].legend(handles, labels)
axs[4,0].set_xlabel('time [s]')
axs[4,0].set_ylabel('u opt')
axs[4,0].grid()
#  -------------------------------------------------------------------
axs[0,1].plot(tds, dx_nom[0,:].T, color='b', label='nom')
axs[0,1].plot(tds, dx_opt[0,:].T, color='g', linestyle='--', label='opt')
axs[0,1].plot(tds, dx_rollout[0,:].T, color='r', linestyle=':', label='rollout')
handles, labels = axs[0,1].get_legend_handles_labels()
axs[0,1].legend(handles, labels)
axs[0,1].set_ylabel('dx0')
axs[0,1].grid()
#  -------------------------------------------------------------------
axs[1,1].plot(tds, dx_nom[1,:].T, color='b', label='nom')
axs[1,1].plot(tds, dx_opt[1,:].T, color='g', linestyle='--', label='opt')
axs[1,1].plot(tds, dx_rollout[1,:].T, color='r', linestyle=':', label='rollout')
handles, labels = axs[1,1].get_legend_handles_labels()
axs[1,1].legend(handles, labels)
axs[1,1].set_ylabel('dx1')
axs[1,1].grid()
#  -------------------------------------------------------------------
axs[2,1].plot(tds, dx_nom[2,:].T*(180/np.pi), color='b', label='nom')
axs[2,1].plot(tds, dx_opt[2,:].T*(180/np.pi), color='g', linestyle='--', label='opt')
axs[2,1].plot(tds, dx_rollout[2,:].T*(180/np.pi), color='r', linestyle=':', label='rollout')
handles, labels = axs[2,1].get_legend_handles_labels()
axs[2,1].legend(handles, labels)
axs[2,1].set_ylabel('dx2')
axs[2,1].grid()
#  -------------------------------------------------------------------
axs[3,1].plot(tds, dx_nom[3,:].T*(180/np.pi), color='b', label='nom')
axs[3,1].plot(tds, dx_opt[3,:].T*(180/np.pi), color='g', linestyle='--', label='opt')
axs[3,1].plot(tds, dx_rollout[3,:].T*(180/np.pi), color='r', linestyle=':', label='rollout')
handles, labels = axs[3,1].get_legend_handles_labels()
axs[3,1].legend(handles, labels)
axs[3,1].set_ylabel('dx3')
axs[3,1].grid()
#  -------------------------------------------------------------------
axs[4,1].plot(tds, np.array(cost_opt), color='b')
handles, labels = axs[4,1].get_legend_handles_labels()
axs[4,1].legend(handles, labels)
axs[4,1].set_xlabel('time [s]')
axs[4,1].set_ylabel('cost')
axs[4,1].grid()
#  -------------------------------------------------------------------

# Animation of Nominal Trajectory
#  -------------------------------------------------------------------
if show_anim:
#  -------------------------------------------------------------------
    x_anim = np.array(x_rollout)
    fig, ax = sliding_pack.plots.plot_nominal_traj(x_anim[0,:].T, x_anim[1,:].T)
    # get slider and pusher patches
    slider, pusher, _, _ = sliding_pack.plots.get_patches_for_square_slider_and_cicle_pusher(
            ax, 
            p_pusher_func, 
            R_pusher_func, 
            x_anim,
            a, r_pusher)
    # call the animation
    ani = animation.FuncAnimation(
            fig,
            sliding_pack.plots.animate_square_slider_and_circle_pusher,
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
