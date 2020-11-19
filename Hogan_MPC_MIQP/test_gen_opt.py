## Import Libraries
#  -------------------------------------------------------------------
import numpy as np
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
N_x = 4 # number of state variables
N_u = 3 # number of actions variables
g = 9.81 # gravity acceleration constant in meter per second square
a = 0.09 # side dimension of the square slider in meters
m = 0.827 # mass of the slider in kilo grams
miu_g = 0.35 # coeficient of friction between slider and table
T = 10 # time of the simulation is seconds
freq = 50 # numer of increments per second
r_pusher = 0.01 # radious of the cilindrical pusher in meter
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N = T*freq # total number of iterations
A = a**2 # area of the slider in meter square
# Area integral of norm of the distance for a square:
int_square = lambda a: dblquad(lambda x,y: np.sqrt((x**2) + (y**2)), -a/2, a/2, -a/2, a/2)[0]
int_A = int_square(a)
# Compute f and m max
f_max = miu_g*m*g # limit force in Newton
m_max = miu_g*m*g*int_A/A # limit torque Newton meter
#  -------------------------------------------------------------------

## Define state and control vectors
#  -------------------------------------------------------------------
# x - state vector
# x[0] - x slider CoM position in the global frame
# x[1] - y slider CoM position in the global frame
# x[2] - slider orientation in the global frame
# x[3] - angle of pusher relative to slider
x = cs.SX.sym('x', 4)
# u - control vector
# u[0] - normal force in the local frame
# u[1] - tangential force in the local frame
# u[2] - relative sliding velocity between pusher and slider
u = cs.SX.sym('u', 3)
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

## Define structures for optimization variables and optimization arguments
#  -------------------------------------------------------------------
class OptVars():
    x = None # optimization independent variables
    g = None # optimization equality constraints
    p = None # optimization parameters
    f = None # optimization cost
    discrete = None # flag for indicating integer variables
class OptArgs():
    x0 = None # initial guess for optimization independent varibles
    p = None # parameters
    lbg = None # lower bound for for constraint g
    ubg = None # upper bound for the constraint g
    lbx = None # lower bound for optimization variables
    ubx = None # upper bound for optimization variables
#  -------------------------------------------------------------------

## Generate Nominal Trajectory (line)
#  -------------------------------------------------------------------
# specify plannar traj for line
## x0_nom = np.linspace(0, 0.5, N)
## x1_nom = np.zeros(x0_nom.shape)
# specify plannar traj for circle
s = np.linspace(-np.pi/2, np.pi/2, N)
Amp = 0.25
x0_nom = Amp*np.cos(s)
x1_nom = Amp*np.sin(s) + Amp
#  -------------------------------------------------------------------
# check trajectory
fig_test = plt.figure()
fig_test.canvas.set_window_title('Matplotlib Animation')
ax = fig_test.add_subplot(111, aspect='equal', autoscale_on=False, \
        xlim=(np.min(x0_nom)-0.1,np.max(x0_nom)+0.1), \
        ylim=(np.min(x1_nom)-0.1,np.max(x1_nom)+0.1) \
)
ax.plot(x0_nom, x1_nom, color='red', linewidth=2.0, linestyle='dashed')
ax.plot(x0_nom[0], x1_nom[0], x0_nom[-1], x1_nom[-1], marker='o', color='red')
ax.grid();
ax.set_aspect('equal', 'box')
plt.show()
#sys.exit(1)
#  -------------------------------------------------------------------
# compute diff for plannar traj
dx0_nom = np.diff(x0_nom)
dx1_nom = np.diff(x1_nom)
# compute traj angle
x2_nom = np.arctan2(dx1_nom, dx0_nom);
x2_nom = np.append(x2_nom, x2_nom[-1])
dx2_nom = np.diff(x2_nom)
# specify angle of the pusher relative to slider
x3_nom = np.zeros(x0_nom.shape)
dx3_nom = np.diff(x3_nom)
# stack state and derivative of state
x_nom = np.vstack((x0_nom, x1_nom, x2_nom, x3_nom))
dx_nom = np.vstack((dx0_nom, dx1_nom, dx2_nom, dx3_nom))/freq
#  -------------------------------------------------------------------
u_nom = cs.SX.sym('u_nom', N_u, N-1)
opt = OptVars()
args = OptArgs()
## ---- Initialize variables for optimization problem ---
opt.f = cs.SX(1,1)
W_f = cs.SX(N_x, N_x);
W_f[0,0] = W_f[1,1] = 10.0;
W_f[2,2] = W_f[3,3] = 1.0;
opt.x = []
opt.g = []
args.x0 = []
args.lbx = []
args.ubx = []
for i in range(N-1):
    # define cost
    f_i = f_func(x_nom[:,i], u_nom[:,i])
    err_i = dx_nom[:,i] - f_i
    opt.f += cs.dot(err_i,cs.mtimes(W_f,err_i))
    # define optimization variables
    opt.x.extend(u_nom[:,i].elements())
    # initial condition for opt var
    args.x0 += [0.0, 0.0, 0.0]
    # opt var boundaries
    args.lbx += [-cs.inf, -cs.inf, -cs.inf]
    args.ubx += [cs.inf, cs.inf, cs.inf]
## ---- Create solver ----
prob = {'f': opt.f, 'x': cs.vertcat(*opt.x)}
solver = cs.nlpsol('solver', 'ipopt', prob)
## ---- Solve optimization problem ----
sol = solver(x0=args.x0, lbx=args.lbx, ubx=args.ubx)
u_sol = sol['x']
u_nom = cs.horzcat(u_sol[0::N_u],u_sol[1::N_u],u_sol[2::N_u]).T
#  -------------------------------------------------------------------
ts = np.linspace(0, T, N)
x0 = x_nom[:,0]
print(x0)
#  -------------------------------------------------------------------

# Plot Optimization Results
#  -------------------------------------------------------------------
fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
ax_x = fig.add_subplot(spec[0, 0])
ax_y = fig.add_subplot(spec[0, 1])
ax_ang = fig.add_subplot(spec[1, 0])
ax_fn = fig.add_subplot(spec[1, 1])
#  -------------------------------------------------------------------
ax_x.plot(ts, x_nom[0,:], color='b', label='nom')
handles, labels = ax_x.get_legend_handles_labels()
ax_x.legend(handles, labels)
ax_x.set(xlabel='time [s]', ylabel='position [m]',
               title='Slider CoM x position')
ax_x.grid()
#  -------------------------------------------------------------------
ax_y.plot(ts, x_nom[1,:], color='b', label='nom')
handles, labels = ax_y.get_legend_handles_labels()
ax_y.legend(handles, labels)
ax_y.set(xlabel='time [s]', ylabel='position [m]',
               title='Slider CoM y position')
ax_y.grid()
#  -------------------------------------------------------------------
ax_ang.plot(ts, x_nom[2,:]*(180/np.pi), color='b', label='slider')
ax_ang.plot(ts, x_nom[3,:]*(180/np.pi), color='r', label='pusher')
handles, labels = ax_ang.get_legend_handles_labels()
ax_ang.legend(handles, labels)
ax_ang.set(xlabel='time [s]', ylabel='angles [degrees]',
               title='Angles of pusher and Slider')
ax_ang.grid()
#  -------------------------------------------------------------------
plt.show(block=False)
#sys.exit(1)
#  -------------------------------------------------------------------

# Animation of Nominal Trajectory
#  -------------------------------------------------------------------
# set up the figure and subplot
fig = plt.figure()
fig.canvas.set_window_title('Matplotlib Animation')
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, \
        xlim=(np.min(x0_nom)-0.1,np.max(x0_nom)+0.1), \
        ylim=(np.min(x1_nom)-0.1,np.max(x1_nom)+0.1) \
)
ax.plot(x_nom[0,:], x_nom[1,:], color='red', linewidth=2.0, linestyle='dashed')
ax.plot(x_nom[0,0], x_nom[1,0], x_nom[0,-1], x_nom[1,-1], marker='o', color='red')
ax.grid();
ax.set_aspect('equal', 'box')
ax.set_title('Pusher-Slider Motion Animation')
d0 = np.array(cs.mtimes(R_func(x0),[-a/2, -a/2, 0]).T)[0]
slider = patches.Rectangle(x0[0:2]+d0[0:2], a, a, x0[2])
pusher = patches.Circle(np.array(p_pusher_func(x0)), radius=r_pusher, color='black')
def init():
    ax.add_patch(slider)
    ax.add_patch(pusher)
    return []
    #return slider,pusher
def animate(i, slider, pusher):
    xi = x_nom[:,i]
    # distance between centre of square reference corner
    di = np.array(cs.mtimes(R_func(xi),[-a/2, -a/2, 0]).T)[0]
    # square reference corner
    ci = xi[0:3] + di
    # compute transformation with respect to rotation angle xi[2]
    trans_ax = ax.transData
    coords = trans_ax.transform(ci[0:2])
    trans_i = transforms.Affine2D().rotate_around(coords[0], coords[1], xi[2])
    # Set changes
    slider.set_transform(trans_ax+trans_i)
    slider.set_xy([ci[0], ci[1]])
    pusher.set_center(np.array(p_pusher_func(xi)))
    return []
# call the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, \
        fargs=(slider,pusher,),
        frames=N,
        interval=T,
        blit=True, repeat=False)
## to save animation, uncomment the line below:
## ani.save('sliding_nominal_traj.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
#show the animation
plt.show()
#  -------------------------------------------------------------------
