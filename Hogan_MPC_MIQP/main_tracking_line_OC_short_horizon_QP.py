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
import numpy.matlib as nplib
from scipy.integrate import dblquad 
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.transforms as transforms
import sys
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
T = 10 # time of the simulation is seconds
freq = 50 # numer of increments per second
r_pusher = 0.005 # radious of the cilindrical pusher in meter
N_MPC = 200 # time horizon for the MPC controller
x_init_val = [-0.01, 0.03, 30*(np.pi/180.), 0]
u_init_val = [0.0, 0.0, 0.0]
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N = T*freq # total number of iterations
N_x = 4
N_u = 3
N_var = (N_x+N_u)*N_MPC
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

## Define structures for optimization variables and optimization arguments
#  -------------------------------------------------------------------
class OptVars():
    x = None # optimization independent variables
    g = None # optimization equality constraints
    p = None # optimization parameters
    f = None # optimization cost
class OptArgs():
    x0 = None # initial guess for optimization independent varibles
    p = None # parameters
    lbg = None # lower bound for for constraint g
    ubg = None # upper bound for the constraint g
    lbx = None # lower bound for optimization variables
    ubx = None # upper bound for optimization variables
#  -------------------------------------------------------------------

## Compute Jacobians
#  -------------------------------------------------------------------
A_func = cs.Function('A_func', [x,u], [cs.jacobian(f_func(x,u), x)], ['x', 'u'], ['A'])
B_func = cs.Function('B_func', [x,u], [cs.jacobian(f_func(x,u), u)], ['x', 'u'], ['B'])
#  -------------------------------------------------------------------

## Generate Nominal Trajectory (line)
#  -------------------------------------------------------------------
# constant input and initial state
u_const = cs.SX(N_u,1); u_const[0] = 0.05
#  -------------------------------------------------------------------
t = cs.SX.sym('t'); ts = np.linspace(0, T, N)
dae = {'x':x, 't':t, 'ode': f_func(x, u_const)}
F = cs.integrator('F', 'cvodes', dae, {'grid':ts, 'output_t0':True})
X_nom_val = F(x0=[0, 0, 0, 0])['xf']
U_nom_val = cs.repmat(u_const, 1, N-1)
U_nom_val = np.array(cs.DM(U_nom_val))
#  -------------------------------------------------------------------

## Compute argumens for the entire nominal trajectory
#  -------------------------------------------------------------------
ARGS_NOM = OptArgs()
## ---- Initialize variables for optimization problem ---
ARGS_NOM.lbg = []
ARGS_NOM.ubg = []
ARGS_NOM.lbx = []
ARGS_NOM.ubx = []
ARGS_NOM.p = []
for i in range(N-1):
    ## ---- Dynamic constraints ----
    ARGS_NOM.lbg += [0, 0, 0, 0]
    ARGS_NOM.ubg += [0, 0, 0, 0]
    ## ---- Control constraints ----
    ARGS_NOM.lbg += [-(miu_p*U_nom_val[0,i]+U_nom_val[1,i])]
    ARGS_NOM.ubg += [cs.inf]
    ARGS_NOM.lbg += [-(miu_p*U_nom_val[0,i]-U_nom_val[1,i])]
    ARGS_NOM.ubg += [cs.inf]
    ## ---- Add States to optimization variables ---
    ARGS_NOM.lbx += [-cs.inf, -cs.inf, -cs.inf, -cs.inf]
    ARGS_NOM.ubx += [cs.inf, cs.inf, cs.inf, cs.inf]
    ## ---- Add Actions to optimization variables ---
    # normal vel
    ARGS_NOM.lbx += [-U_nom_val[0,i]]
    ARGS_NOM.ubx += [cs.inf]
    # tangential vel
    ARGS_NOM.lbx += [-cs.inf]
    ARGS_NOM.ubx += [cs.inf]
    # relative sliding vel
    ARGS_NOM.lbx += [U_nom_val[2,i]]
    ARGS_NOM.ubx += [U_nom_val[2,i]]
    ## ---- Set nominal trajectory as parameters ----
    ARGS_NOM.p.extend(X_nom_val[:,i].elements())
    ARGS_NOM.p.extend(U_nom_val[:,i])
## ---- Add last States to optimization variables ---
ARGS_NOM.lbx += [-cs.inf, -cs.inf, -cs.inf, -cs.inf]
ARGS_NOM.ubx += [cs.inf, cs.inf, cs.inf, cs.inf]
ARGS_NOM.p.extend(X_nom_val[:,-1].elements())
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
opt = OptVars()
## ---- Set optimization objective ----------
Qcost = cs.diag(cs.SX([3.0,3.0,0.01,0]))
Rcost = cs.diag(cs.SX([1,1,0.0]))
opt.f = cs.dot(X_bar[:,-1],cs.mtimes(Qcost,X_bar[:,-1]))
for i in range(N_MPC-1):
    opt.f += cs.dot(X_bar[:,i],cs.mtimes(Qcost,X_bar[:,i])) + cs.dot(U_bar[:,i],cs.mtimes(Rcost,U_bar[:,i]))
## ---- Set optimization variables ----
opt.x = []
for i in range(N_MPC-1):
    opt.x.extend(X_bar[:,i].elements())
    opt.x.extend(U_bar[:,i].elements())
opt.x.extend(X_bar[:,-1].elements())
## ---- Set optimzation constraints ----
opt.g = []
opt.g.extend([X_bar[:,0]+X_nom[:,0]-x_init]) ## Initial Conditions
for i in range(N_MPC-1):
    ## Dynamic constraints
    Ai = A_func(X_nom[:,i], U_nom[:,i])
    Bi = B_func(X_nom[:,i], U_nom[:,i])
    opt.g.extend([X_bar[:,i+1]-X_bar[:,i]-h*(cs.mtimes(Ai,X_bar[:,i])+cs.mtimes(Bi,U_bar[:,i]))])
    ## Control constraints
    opt.g += [miu_p*U_bar[0,i]+U_bar[1,i]]
    opt.g += [miu_p*U_bar[0,i]-U_bar[1,i]]
## ---- Set optimization parameters ----
opt.p = []
opt.p.extend(x_init.elements())
opt.p.extend(u_init.elements())
for i in range(N_MPC-1):
    opt.p.extend(X_nom[:,i].elements())
    opt.p.extend(U_nom[:,i].elements())
opt.p.extend(X_nom[:,-1].elements())
## ---- Create solver ----
prob = {'f': opt.f, 'x': cs.vertcat(*opt.x), 'g': cs.vertcat(*opt.g), 'p': cs.vertcat(*opt.p)}
solver = cs.nlpsol('solver', 'ipopt', prob)
#solver = cs.nlpsol('solver', 'snopt', prob)
#solver = cs.qpsol('S', 'qpoases', prob, {'sparse':True})
#solver = cs.qpsol('solver', 'gurobi', prob)
#  -------------------------------------------------------------------

## Set arguments and solve
#  -------------------------------------------------------------------
args = OptArgs()
# Indexing
idx = 0
idx_x_i = idx*(N_x+N_u)
idx_x_f = (idx+N_MPC-1)*(N_x+N_u)+N_x
# warm start
args.x0 = ARGS_NOM.p[idx_x_i:idx_x_f]
# optimization bounderies
args.lbx = ARGS_NOM.lbx[idx_x_i:idx_x_f]
args.ubx = ARGS_NOM.ubx[idx_x_i:idx_x_f]
## parameters
args.p = []
args.p.extend(x_init_val)
args.p.extend(u_init_val)
args.p.extend(ARGS_NOM.p[idx_x_i:idx_x_f])
## constraints bounderies
idx_g_i = idx*6
idx_g_f = (idx+N_MPC-1)*6
args.lbg = []
args.ubg = []
# initial state constraint
args.lbg += [0, 0, 0, 0]
args.ubg += [0, 0, 0, 0]
# dynamics and friction constraints
args.lbg.extend(ARGS_NOM.lbg[idx_g_i:idx_g_f])
args.ubg.extend(ARGS_NOM.ubg[idx_g_i:idx_g_f])
## ---- Solve the optimization ----
sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx, lbg=args.lbg, ubg=args.ubg, p=args.p)
x_opt = sol['x']
## ---- Compute actual trajectory and controls ----
X_bar_opt = np.array(cs.horzcat(x_opt[0::7],x_opt[1::7],x_opt[2::7],x_opt[3::7]).T)
U_bar_opt = np.array(cs.horzcat(x_opt[4::7],x_opt[5::7],x_opt[6::7]).T)
X_bar_opt = np.array(X_bar_opt)
U_bar_opt = np.array(U_bar_opt)
X_opt = X_bar_opt + X_nom_val[:,idx:(idx+N_MPC)]
U_opt = U_bar_opt + U_nom_val[:,idx:(idx+N_MPC-1)]
#sys.exit(1)
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
X_nom_val = np.array(X_nom_val)
X_opt = np.array(X_opt)
#  -------------------------------------------------------------------
fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
ax_x = fig.add_subplot(spec[0, 0])
ax_y = fig.add_subplot(spec[0, 1])
ax_ang = fig.add_subplot(spec[1, 0])
ax_fn = fig.add_subplot(spec[1, 1])
#  -------------------------------------------------------------------
ax_x.plot(ts, X_nom_val[0,:], color='b', label='nom')
ax_x.plot(ts[0:N_MPC], X_opt[0,:], color='r', label='opt')
handles, labels = ax_x.get_legend_handles_labels()
ax_x.legend(handles, labels)
ax_x.set(xlabel='time [s]', ylabel='position [m]',
               title='Slider CoM x position')
ax_x.grid()
#  -------------------------------------------------------------------
ax_y.plot(ts, X_nom_val[1,:], color='b', label='nom')
ax_y.plot(ts[0:N_MPC], X_opt[1,:], color='r', label='opt')
handles, labels = ax_y.get_legend_handles_labels()
ax_y.legend(handles, labels)
ax_y.set(xlabel='time [s]', ylabel='position [m]',
               title='Slider CoM y position')
ax_y.grid()
#  -------------------------------------------------------------------
ax_ang.plot(ts[0:N_MPC], X_opt[2,:]*(180/np.pi), color='b', label='slider')
ax_ang.plot(ts[0:N_MPC], X_opt[3,:]*(180/np.pi), color='r', label='pusher')
handles, labels = ax_ang.get_legend_handles_labels()
ax_ang.legend(handles, labels)
ax_ang.set(xlabel='time [s]', ylabel='angles [degrees]',
               title='Angles of pusher and Slider')
ax_ang.grid()
#  -------------------------------------------------------------------
ax_fn.plot(ts[0:N_MPC-1], U_opt[0,:], color='b', label='norm')
ax_fn.plot(ts[0:N_MPC-1], U_opt[1,:], color='g', label='tan')
handles, labels = ax_fn.get_legend_handles_labels()
ax_fn.legend(handles, labels)
ax_fn.set(xlabel='time [s]', ylabel='force [N]',
               title='Pusher vel. on slider')
ax_fn.grid()
#  -------------------------------------------------------------------
plt.show(block=False)
#sys.exit(1)
#  -------------------------------------------------------------------

# Animation of Nominal Trajectory
#  -------------------------------------------------------------------
# set up the figure and subplot
fig_ani = plt.figure()
fig_ani.canvas.set_window_title('Matplotlib Animation')
ax_ani = fig_ani.add_subplot(111, aspect='equal', autoscale_on=False, \
        xlim=(-0.1,0.6), ylim=(-0.1,0.1) \
)
# draw nominal trajectory
ax_ani.plot(X_nom_val[0,:], X_nom_val[1,:], color='red', linewidth=2.0, linestyle='dashed')
ax_ani.plot(X_nom_val[0,0], X_nom_val[1,0], X_nom_val[0,-1], X_nom_val[1,-1], marker='o', color='red')
ax_ani.grid();
#ax_ani.set_axisbelow(True)
ax_ani.set_aspect('equal', 'box')
ax_ani.set_title('Pusher-Slider Motion Animation')
slider = patches.Rectangle([0,0], a, a)
pusher = patches.Circle([0,0], radius=r_pusher, color='black')
def init():
    ax_ani.add_patch(slider)
    ax_ani.add_patch(pusher)
    return []
    #return slider,
def animate(i, slider, pusher):
    xi = X_opt[:,i]
    # distance between centre of square reference corner
    di=np.array(cs.mtimes(R_pusher_func(xi),[-a/2, -a/2, 0]).T)[0]
    # square reference corner
    ci = xi[0:3] + di
    # compute transformation with respect to rotation angle xi[2]
    trans_ax = ax_ani.transData
    coords = trans_ax.transform(ci[0:2])
    trans_i = transforms.Affine2D().rotate_around(coords[0], coords[1], xi[2])
    # Set changes
    #slider.set_transform(trans_ax+trans_i)
    slider.set_transform(trans_ax+trans_i)
    slider.set_xy([ci[0], ci[1]])
    pusher.set_center(np.array(p_pusher_func(xi)))
    return []
#init()
# call the animation
ani = animation.FuncAnimation(fig_ani, animate, init_func=init, \
        fargs=(slider,pusher,),
        frames=N_MPC,
        interval=T,
        blit=True, repeat=False)
## to save animation, uncomment the line below:
#ani.save('sliding_tracking_line_fullTO_QP.mp4', fps=freq, extra_args=['-vcodec', 'libx264'])
#ani.save('sliding_tracking_line_fullTO_QP.gif', writer='imagemagick', fps=freq)
#show the animation
plt.show()
#  -------------------------------------------------------------------
