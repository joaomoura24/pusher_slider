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
g = 9.81 # gravity acceleration constant in meter per second square
a = 0.09 # side dimension of the square slider in meters
m = 0.827 # mass of the slider in kilo grams
miu_g = 0.35 # coeficient of friction between slider and table
miu_p = 0.3 # coeficient of friction between pusher and slider
T = 12 # time of the simulation is seconds
freq = 50 # numer of increments per second
r_pusher = 0.005 # radious of the cilindrical pusher in meter
N_MPC = 150 # time horizon for the MPC controller
# N_MPC = 35 # time horizon for the MPC controller
x_init_val = [-0.01, 0.03, 30*(np.pi/180.), 0]
u_init_val = [0.0, 0.0, 0.0]
# solver_name = 'ipopt'
# solver_name = 'snopt'
solver_name = 'gurobi'
# solver_name = 'qpoases'
opts_dict = {'print_time': 0}
no_printing = True
code_gen = False
#  -------------------------------------------------------------------
## get string name
prog_name = 'MPC' + '_TH' + str(N_MPC) + '_' + solver_name + '_codeGen_' + str(code_gen)
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N = T*freq # total number of iterations
dt = 1.0/freq # sampling time
N_x = 4
N_u = 3
h = 1./freq # time interval of each iteration
A = a**2 # area of the slider in meter square
f_max = miu_g*m*g # limit force in Newton
# Area integral of norm of the distance for a square:
int_square = lambda a: dblquad(lambda x,y: np.sqrt(x**2 + y**2), -a/2, a/2, -a/2, a/2)[0]
int_A = int_square(a)
m_max = miu_g*m*g*int_A/A # limit torque Newton meter
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
x0_nom, x1_nom = my_trajectories.generate_traj_line(0.5, 0.0, N)
# x0_nom, x1_nom = my_trajectories.generate_traj_line(0.5, 0.3, N)
# x0_nom, x1_nom = my_trajectories.generate_traj_circle(-np.pi/2, 3*np.pi/2, 0.25, N)
# x0_nom, x1_nom = my_trajectories.generate_traj_eight(0.5, N)
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
#  -------------------------------------------------------------------

## Set up QP Optimization Problem
#  -------------------------------------------------------------------
## ---- Define optimization objective ----------
Qcost = cs.diag(cs.SX([3.0,3.0,0.01,0]))
Rcost = cs.diag(cs.SX([1,1,0.0]))
cost_f = cs.Function('cost', [x, u], [cs.dot(x,cs.mtimes(Qcost,x)) + cs.dot(u,cs.mtimes(Rcost,u))])
cost_F = cost_f.map(N_MPC-1)
## ---- Initialize optimization and argument variables ---
opt = my_opt.OptVars()
## ---- cost function ----
opt.f = cs.sum2(cost_F(X_bar[:,0:-1], U_bar)) + cost_f(X_bar[:,-1], cs.SX(N_u, 1)) 
# opt.f = cs.dot(X_bar[:,-1],cs.mtimes(Qcost,X_bar[:,-1]))
# for i in range(N_MPC-1):
#     opt.f += cs.dot(X_bar[:,i],cs.mtimes(Qcost,X_bar[:,i])) + cs.dot(U_bar[:,i],cs.mtimes(Rcost,U_bar[:,i]))
## ---- Set optimization variables ----
opt.x = []
for i in range(N_MPC-1):
    opt.x += X_bar[:,i].elements()
    opt.x += U_bar[:,i].elements()
opt.x += X_bar[:,-1].elements()
## ---- Set optimzation constraints ----
opt.g = []
opt.g += (X_bar[:,0]+X_nom[:,0]-x_init).elements() ## Initial Conditions
for i in range(N_MPC-1):
    ## Dynamic constraints
    opt.g += dyn_err_f(X_nom[:,i], U_nom[:,i], X_bar[:,i], X_bar[:,i+1], U_bar[:,i]).elements()
    ## Control constraints
    opt.g += fric_cone_c(U_bar[:,i]).elements()
## ---- Set optimization parameters ----
opt.p = x_init.elements()
opt.p += u_init.elements()
for i in range(N_MPC-1):
    opt.p += X_nom[:,i].elements()
    opt.p += U_nom[:,i].elements()
opt.p += X_nom[:,-1].elements()
## ---- Set solver options ----
if solver_name == 'ipopt':
    if no_printing: opts_dict['ipopt.print_level'] = 0
if solver_name == 'snopt':
    if no_printing: opts_dict['snopt'] = {'Major print level': '0', 'Minor print level': '0'}
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
Nidx = N-N_MPC
X_plot = np.empty([N_x, Nidx+1])
U_plot = np.empty([N_u, Nidx])
X_plot[:,0] = x_init_val
X_future = np.empty([N_x, N_MPC, Nidx])
comp_time = np.empty(Nidx)
#  -------------------------------------------------------------------

## Set arguments and solve
#  -------------------------------------------------------------------
x0 = x_init_val
u0 = u_init_val
args = my_opt.OptArgs()
for idx in range(Nidx):
    # Indexing
    idx_x_i = idx*(N_x+N_u)
    idx_x_f = (idx+N_MPC-1)*(N_x+N_u)+N_x
    idx_g_i = idx*6
    idx_g_f = (idx+N_MPC-1)*6
    # warm start
    if idx==0:
        args.x0 = ARGS_NOM.p[idx_x_i:idx_x_f]
    else:
        args.x0 = x_opt[6:-1].elements()
        args.x0 += ARGS_NOM.p[(idx_x_f-(N_u+N_x)):idx_x_f]
    # setting optimization bounderies from nominal traj
    args.lbx = ARGS_NOM.lbx[idx_x_i:idx_x_f]
    args.ubx = ARGS_NOM.ubx[idx_x_i:idx_x_f]
    ## setting parameters
    args.p = [] # set to empty before reinitialize
    args.p += x0
    args.p += u0
    args.p += ARGS_NOM.p[idx_x_i:idx_x_f]
    # initial state constraint
    args.lbg = [0]*N_x
    args.ubg = [0]*N_x
    # dynamics and friction constraints
    args.lbg += ARGS_NOM.lbg[idx_g_i:idx_g_f]
    args.ubg += ARGS_NOM.ubg[idx_g_i:idx_g_f]
    ## ---- Solve the optimization ----
    start_time = time.time()
    sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg, p=args.p)
    x_opt = sol['x']
    # save computation time
    comp_time[idx] = time.time() - start_time
    ## ---- Compute actual trajectory and controls ----
    X_bar_opt = cs.horzcat(x_opt[0::7],x_opt[1::7],x_opt[2::7],x_opt[3::7]).T
    U_bar_opt = cs.horzcat(x_opt[4::7],x_opt[5::7],x_opt[6::7]).T
    X_opt = X_bar_opt + X_nom_val[:,idx:(idx+N_MPC)]
    U_opt = U_bar_opt + U_nom_val[:,idx:(idx+N_MPC-1)]
    ## ---- Update initial conditions ----
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
ts = np.linspace(0, T, N)
ts_X = ts[0:Nidx+1]
ts_U = ts[0:Nidx]
ts_opt = ts[0:N_MPC]
X_nom_val = np.array(X_nom_val)
#  -------------------------------------------------------------------
fig = plt.figure(constrained_layout=True, figsize=(6, 8))
spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
ax_x = fig.add_subplot(spec[0, 0])
ax_y = fig.add_subplot(spec[0, 1])
ax_ang = fig.add_subplot(spec[1, 0])
ax_fn = fig.add_subplot(spec[1, 1])
ax_t = fig.add_subplot(spec[2, :])
#  -------------------------------------------------------------------
ax_x.plot(ts, X_nom_val[0,:], color='b', label='nom')
ax_x.plot(ts_X, X_plot[0,:], color='r', label='opt')
handles, labels = ax_x.get_legend_handles_labels()
ax_x.legend(handles, labels)
ax_x.set(xlabel='time [s]', ylabel='position [m]',
               title='Slider CoM x position')
ax_x.grid()
#  -------------------------------------------------------------------
ax_y.plot(ts, X_nom_val[1,:], color='b', label='nom')
ax_y.plot(ts_X, X_plot[1,:], color='r', label='opt')
handles, labels = ax_y.get_legend_handles_labels()
ax_y.legend(handles, labels)
ax_y.set(xlabel='time [s]', ylabel='position [m]',
               title='Slider CoM y position')
ax_y.grid()
#  -------------------------------------------------------------------
ax_ang.plot(ts_X, X_plot[2,:]*(180/np.pi), color='b', label='slider')
ax_ang.plot(ts_X, X_plot[3,:]*(180/np.pi), color='r', label='pusher')
handles, labels = ax_ang.get_legend_handles_labels()
ax_ang.legend(handles, labels)
ax_ang.set(xlabel='time [s]', ylabel='angles [degrees]',
               title='Angles of pusher and Slider')
ax_ang.grid()
#  -------------------------------------------------------------------
ax_fn.plot(ts_U, U_plot[0,:], color='b', label='norm')
ax_fn.plot(ts_U, U_plot[1,:], color='g', label='tan')
handles, labels = ax_fn.get_legend_handles_labels()
ax_fn.legend(handles, labels)
ax_fn.set(xlabel='time [s]', ylabel='force [N]',
               title='Pusher vel. on slider')
ax_fn.grid()
#  -------------------------------------------------------------------
ax_t.plot(ts_U, comp_time)
ax_y.set(xlabel='time [s]', ylabel='time [s]',
               title='Computational time')
ax_t.grid()
#  -------------------------------------------------------------------
plt.show(block=False)
#  -------------------------------------------------------------------

# Animation of Nominal Trajectory
#  -------------------------------------------------------------------
# set up the figure and subplot
fig_ani = plt.figure()
fig_ani.canvas.set_window_title('Matplotlib Animation')
ax_ani = fig_ani.add_subplot(111, aspect='equal', autoscale_on=False, \
        xlim=(-0.05,0.65), ylim=(-0.15,0.15) \
)
# draw nominal trajectory
ax_ani.plot(X_nom_val[0,:], X_nom_val[1,:], color='red', linewidth=2.0, linestyle='dashed')
ax_ani.plot(X_nom_val[0,0], X_nom_val[1,0], X_nom_val[0,-1], X_nom_val[1,-1], marker='o', color='red')
ax_ani.grid();
ax_ani.set_aspect('equal', 'box')
ax_ani.set_title('Pusher-Slider Motion Animation')
slider = patches.Rectangle([0,0], a, a)
pusher = patches.Circle([0,0], radius=r_pusher, color='black')
# Plot centre of the slider
#path_future, = ax_ani.plot(X_future[0,:,0], X_future[1,:,0], color='orange', linestyle='dashed')
path_future, = ax_ani.plot(x_init_val[0], x_init_val[1], color='orange', linestyle='dashed')
path_past, = ax_ani.plot(x_init_val[0], x_init_val[1], color='orange')
path_past.set_linewidth(2)
def init():
    ax_ani.add_patch(slider)
    ax_ani.add_patch(pusher)
    return []
    #return slider,
def animate(i, slider, pusher):
    xi = X_plot[:,i]
    # distance between centre of square reference corner
    di=np.array(cs.mtimes(R_pusher_func(xi),[-a/2, -a/2, 0]).T)[0]
    # square reference corner
    ci = xi[0:3] + di
    # compute transformation with respect to rotation angle xi[2]
    trans_ax = ax_ani.transData
    coords = trans_ax.transform(ci[0:2])
    trans_i = transforms.Affine2D().rotate_around(coords[0], coords[1], xi[2])
    # Plot centre of the slider
    path_future.set_data(X_future[0,:,i], X_future[1,:,i])
    path_past.set_data(X_plot[0,0:i], X_plot[1,0:i])
    # Set changes
    slider.set_transform(trans_ax+trans_i)
    slider.set_xy([ci[0], ci[1]])
    pusher.set_center(np.array(p_pusher_func(xi)))
    return []
#init()
# call the animation
ani = animation.FuncAnimation(fig_ani, animate, init_func=init, \
        fargs=(slider,pusher,),
        frames=Nidx,
        interval=T,
        blit=True, repeat=False)
## to save animation, uncomment the line below:
#ani.save('mpc_TH50.mp4', fps=freq, extra_args=['-vcodec', 'libx264'])
#ani.save('sliding_tracking_line_fullTO_QP.gif', writer='imagemagick', fps=freq)
#show the animation
plt.show()
#  -------------------------------------------------------------------
