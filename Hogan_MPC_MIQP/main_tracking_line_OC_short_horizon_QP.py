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
#  -------------------------------------------------------------------

## Build Motion Model
#  -------------------------------------------------------------------
c = m_max/f_max
L = cs.SX.sym('L', cs.Sparsity.diag(3))
L[0,0] = L[1,1] = 1; L[2,2] = 1/(c**2);
ctheta = cs.cos(x[2]); stheta = cs.sin(x[2])
R = cs.SX(3,3)
R[0,0] = ctheta; R[0,1] = -stheta; R[1,0] = stheta; R[1,1] = ctheta; R[2,2] = 1;
R_func = cs.Function('R', [x], [R])
xc = -a/2; yc = (a/2)*cs.sin(x[3])
Jc = cs.SX(2,3)
Jc[0,0] = 1; Jc[1,1] = 1; Jc[0,2] = -yc; Jc[1,2] = xc;
B = cs.SX(Jc.T)
#  -------------------------------------------------------------------
rc = cs.SX(2,1); rc[0] = xc-r_pusher; rc[1] = yc
p_pusher = cs.mtimes(R[0:2,0:2], rc)[0:2] + x[0:2]
p_pusher_func = cs.Function('p_pusher', [x], [p_pusher])
#  -------------------------------------------------------------------
f = cs.SX(cs.vertcat(cs.mtimes(cs.mtimes(R,L),cs.mtimes(B,u[0:2])),u[2]))
f_func = cs.Function('f', [x,u], [f])
#  -------------------------------------------------------------------

## Compute Jacobians
#  -------------------------------------------------------------------
A = cs.jacobian(f, x)#[0:3,0:3]
A_func = cs.Function('A', [x,u], [A])
B = cs.jacobian(f, u)#[0:3,0:2]
B_func = cs.Function('B', [x,u], [B])
#  -------------------------------------------------------------------

## Generate Nominal Trajectory (line)
#  -------------------------------------------------------------------
# constant input and initial state
u_const = cs.SX(N_u,1); u_const[0] = 0.05
#  -------------------------------------------------------------------
t = cs.SX.sym('t'); ts = np.linspace(0, T, N)
dae = {'x':x, 't':t, 'ode': f_func(x, u_const)}
F = cs.integrator('F', 'cvodes', dae, {'grid':ts, 'output_t0':True})
X_nom = F(x0=[0, 0, 0, 0])['xf']
U_nom = cs.repmat(u_const, 1, N)
#  -------------------------------------------------------------------

## Set up QP Optimization Problem
#  -------------------------------------------------------------------
# Set nominal trajectory to numerical values
#X_nom = np.array(X_nom[:,0:N_MPC])
X_nom = X_nom[:,0:N_MPC]
U_nom = np.array(cs.DM(U_nom[:,0:N_MPC]))
## ---- Input variables ---
X_bar = cs.SX.sym('x_bar', 4, N_MPC)
U_bar = cs.SX.sym('u_bar', 3, N_MPC)
P = cs.SX.sym('p', 4)
## ---- Optimization objective ----------
Qcost = cs.diag(cs.SX([3.0,3.0,0.01,0]))
Rcost = cs.diag(cs.SX([1,1,0.0]))
Cost = cs.dot(X_bar[:,-1],cs.mtimes(Qcost,X_bar[:,-1]))
for i in range(N_MPC-1):
    Cost += cs.dot(X_bar[:,i],cs.mtimes(Qcost,X_bar[:,i])) + cs.dot(U_bar[:,i],cs.mtimes(Rcost,U_bar[:,i]))
## ---- Define optimization variables ----
class OptVars():
    x = None # optimization independent variables
    g = None # optimization equality constraints
    p = None # optimization parameters
opt = OptVars()
## ---- Set optimization variables ----
opt.x = cs.reshape(cs.vertcat(X_bar,U_bar),1,N_var)[0:(N_var-N_u)]
## ---- Define optimization arguments ----
class OptArgs():
    x0 = None # initial guess for optimization independent varibles
args = OptArgs()
## ---- Set optimization arguments ----
args.x0 = cs.reshape(cs.vertcat(X_nom,U_nom),1,N_var)[0:(N_var-N_u)]
## ---- Initialize variables for optimization problem ---
lbw = []
ubw = []
g = []
lbg = []
ubg = []
## ---- Initial Conditions ----
g += [X_bar[:,0]+X_nom[:,0]-P]
lbg += [0, 0, 0, 0]
ubg += [0, 0, 0, 0]
for i in range(N_MPC-1):
    ## ---- Dynamic constraints ----
    Ai = A_func(X_nom[:,i], U_nom[:,i])
    Bi = B_func(X_nom[:,i], U_nom[:,i])
    g += [X_bar[:,i+1]-X_bar[:,i]-h*(cs.mtimes(Ai,X_bar[:,i])+cs.mtimes(Bi,U_bar[:,i]))]
    lbg += [0, 0, 0, 0]
    ubg += [0, 0, 0, 0]
for i in range(N_MPC-1):
    ## ---- Control constraints ----
    g += [miu_p*U_bar[0,i]+U_bar[1,i]]
    lbg += [-(miu_p*U_nom[0,i]+U_nom[1,i])]
    ubg += [cs.inf]
    g += [miu_p*U_bar[0,i]-U_bar[1,i]]
    lbg += [-(miu_p*U_nom[0,i]-U_nom[1,i])]
    ubg += [cs.inf]
for i in range(N_MPC-1):
    ## ---- Add States to optimization variables ---
    lbw += [-cs.inf, -cs.inf, -cs.inf, -cs.inf]
    ubw += [cs.inf, cs.inf, cs.inf, cs.inf]
    ## ---- Add Actions to optimization variables ---
    # actions
    # normal force
    lbw += [-U_nom[0,i]]
    ubw += [cs.inf]
    # tangential force
    lbw += [-cs.inf]
    ubw += [cs.inf]
    # relative sliding velocity
    lbw += [U_nom[2,i]]
    ubw += [U_nom[2,i]]
## ---- Add last States to optimization variables ---
lbw += [-cs.inf, -cs.inf, -cs.inf, -cs.inf]
ubw += [cs.inf, cs.inf, cs.inf, cs.inf]
## ---- Create solver ----
prob = {'f': Cost, 'x': opt.x, 'g': cs.vertcat(*g), 'p': P}
#solver = cs.nlpsol('solver', 'ipopt', prob)
#solver = cs.nlpsol('solver', 'snopt', prob)
#solver = cs.qpsol('S', 'qpoases', prob, {'sparse':True})
solver = cs.qpsol('solver', 'gurobi', prob)
## ---- Set the appropriate parameters ----
x0_i = [-0.01, 0.03, 30*(np.pi/180.), 0]
## ---- Solve optimization problem ----
sol = solver(x0=args.x0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=x0_i)
w_opt = sol['x']
## ---- Compute actual trajectory and controls ----
X_bar_opt = cs.vertcat(w_opt[0::7],w_opt[1::7],w_opt[2::7],w_opt[3::7])
U_bar_opt = cs.vertcat(w_opt[4::7],w_opt[5::7],w_opt[6::7])
X_bar_opt = np.array(X_bar_opt)
U_bar_opt = np.array(U_bar_opt)
X_opt = X_bar_opt + X_nom
U_opt = U_bar_opt + U_nom[:,0:N_MPC-1]
#sys.exit(1)
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
X_nom = np.array(X_nom)
X_opt = np.array(X_opt)
#  -------------------------------------------------------------------
fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
ax_x = fig.add_subplot(spec[0, 0])
ax_y = fig.add_subplot(spec[0, 1])
ax_ang = fig.add_subplot(spec[1, 0])
ax_fn = fig.add_subplot(spec[1, 1])
#  -------------------------------------------------------------------
ax_x.plot(ts[0:N_MPC], X_nom[0,:], color='b', label='nom')
ax_x.plot(ts[0:N_MPC], X_opt[0,:], color='r', label='opt')
handles, labels = ax_x.get_legend_handles_labels()
ax_x.legend(handles, labels)
ax_x.set(xlabel='time [s]', ylabel='position [m]',
               title='Slider CoM x position')
ax_x.grid()
#  -------------------------------------------------------------------
ax_y.plot(ts[0:N_MPC], X_nom[1,:], color='b', label='nom')
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
ax_ani.plot(X_nom[0,:], X_nom[1,:], color='red', linewidth=2.0, linestyle='dashed')
ax_ani.plot(X_nom[0,0], X_nom[1,0], X_nom[0,-1], X_nom[1,-1], marker='o', color='red')
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
    di=np.array(cs.mtimes(R_func(xi),[-a/2, -a/2, 0]).T)[0]
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
