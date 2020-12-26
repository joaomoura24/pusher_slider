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
T = 5 # time of the simulation is seconds
freq = 50 # numer of increments per second
r_pusher = 0.01 # radious of the cilindrical pusher in meter
show_anim = True
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N = T*freq # total number of iterations
dt = 1.0/freq
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
p_pusher_func = cs.Function('p_pusher_func', [x], [my_dynamics.square_slider_quasi_static_ellipsoidal_limit_surface_p(x, beta)])
#  -------------------------------------------------------------------
f_func = cs.Function('f_func', [x,u], [my_dynamics.square_slider_quasi_static_ellipsoidal_limit_surface_f(x,u, beta)])
#  -------------------------------------------------------------------

## Generate Nominal Trajectory
#  -------------------------------------------------------------------
# x0_nom, x1_nom = my_trajectories.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.25, N)
# x0_nom, x1_nom = my_trajectories.generate_traj_line(0.5, 0.3, N)
x0_nom, x1_nom = my_trajectories.generate_traj_eight(0.5, N)
#  ------------------------------------------------------------------
# compute diff for plannar traj
x_nom, dx_nom = my_trajectories.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  ------------------------------------------------------------------
# stack state and derivative of state
x_nom, dx_nom = my_trajectories.compute_nomState_from_nomTraj(x0_nom, x1_nom, dt)
#  -------------------------------------------------------------------

## Initialize variables for optimization problem
#  -------------------------------------------------------------------
# control path variables
u_nom = cs.SX.sym('u_nom', N_u, N-1)
#  -------------------------------------------------------------------
# declare cost functiopn
W_f = cs.diag(cs.SX([10.0,10.0,1.0,1.0]))
vel_error = dx - f_func(x, u)
cost_f = cs.Function('cost', [x, dx, u], [cs.dot(vel_error,cs.mtimes(W_f,vel_error))])
cost_F = cost_f.map(N-1)
#  -------------------------------------------------------------------
opt = my_opt.OptVars()
# define cost function
opt.f = cs.sum2(cost_F(x_nom[:,-2], dx_nom, u_nom))
# define optimization variables
opt.x = cs.vertcat(*u_nom.elements())
#  -------------------------------------------------------------------

## Generating solver
#  -------------------------------------------------------------------
prob = {'f': opt.f, 'x': opt.x}
solver = cs.nlpsol('solver', 'ipopt', prob)
#  -------------------------------------------------------------------

## Instanciating optimizer arguments
#  -------------------------------------------------------------------
args = my_opt.OptArgs()
# initial condition for opt var
args.x0 = [0.0]*((N-1)*3)
# opt var boundaries
args.lbx = [-cs.inf]*((N-1)*3)
args.ubx = [cs.inf]*((N-1)*3)
#  -------------------------------------------------------------------

## Solve optimization problem
#  -------------------------------------------------------------------
sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx)
u_sol = sol['x']
u_opt = np.array(cs.horzcat(u_sol[0::N_u],u_sol[1::N_u],u_sol[2::N_u]).T)
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6,7))
#  -------------------------------------------------------------------
ts = np.linspace(0, T, N)
axs[0].plot(ts, x_nom[0,:], color='b', label='x')
axs[0].plot(ts, x_nom[1,:], color='g', label='y')
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels)
axs[0].set_ylabel('pos [m]')
axs[0].set_title('Slider CoM')
axs[0].grid()
#  -------------------------------------------------------------------
axs[1].plot(ts, x_nom[2,:]*(180/np.pi), color='b', label='slider')
axs[1].plot(ts, x_nom[3,:]*(180/np.pi), color='r', label='pusher')
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles, labels)
axs[1].set_ylabel('angles [deg]')
axs[1].set_title('Angles of pusher and slider')
axs[1].grid()
#  -------------------------------------------------------------------
ts = np.linspace(0, T, N-1)
axs[2].plot(ts, u_opt[0,:], color='b', label='norm')
axs[2].plot(ts, u_opt[1,:], color='r', label='tan')
handles, labels = axs[2].get_legend_handles_labels()
axs[2].legend(handles, labels)
axs[2].set_xlabel('time [s]')
axs[2].set_ylabel('vel [m/s]')
axs[2].set_title('Puhser control vel')
axs[2].grid()
#  -------------------------------------------------------------------

# Animation of Nominal Trajectory
#  -------------------------------------------------------------------
if show_anim:
#  -------------------------------------------------------------------
    fig, ax = my_plots.plot_nominal_traj(x0_nom, x1_nom)
    # get slider and pusher patches
    slider, pusher = my_plots.get_patches_for_square_slider_and_cicle_pusher(
            ax, 
            p_pusher_func, 
            R_pusher_func, 
            x_nom,
            a, r_pusher)
    # call the animation
    ani = animation.FuncAnimation(
            fig,
            my_plots.animate_square_slider_and_circle_pusher,
            fargs=(slider, pusher, ax, p_pusher_func, R_pusher_func, x_nom, a),
            frames=N,
            interval=T,
            blit=True,
            repeat=False)
    ## to save animation, uncomment the line below:
    ## ani.save('sliding_nominal_traj.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
#  -------------------------------------------------------------------

#  -------------------------------------------------------------------
plt.show()
#  -------------------------------------------------------------------
