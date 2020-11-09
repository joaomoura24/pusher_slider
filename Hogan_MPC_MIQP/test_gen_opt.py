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

## Generate Nominal Trajectory (line)
#  -------------------------------------------------------------------
# constant input and initial state
u_const = [0.05, 0.0, 0.0]
x0 = [0, 0, 0*(np.pi/180), 0]
#  -------------------------------------------------------------------
t = cs.SX.sym('t')
dae = {'x':x, 't':t, 'ode': f_func(x, u_const)}
ts = np.linspace(0, T, N)
F = cs.integrator('F', 'cvodes', dae, {'grid':ts, 'output_t0':True})
sol = F(x0=x0)
x_nom = np.array(F(x0=x0)['xf'])
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
        xlim=(-0.1,0.6), ylim=(-0.1,0.1) \
)
ax.plot(x_nom[0,:], x_nom[1,:], color='red', linewidth=2.0, linestyle='dashed')
ax.plot(x_nom[0,0], x_nom[1,0], x_nom[0,-1], x_nom[1,-1], marker='o', color='red')
ax.grid();
#ax.set_axisbelow(True)
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
