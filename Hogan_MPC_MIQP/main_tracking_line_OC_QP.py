## Author: Joao Moura
## Date: 21/08/2020
#  -------------------------------------------------------------------
## Description:
#  This script implements a quadratic programming (QP) optimal controller (OC)
#  for tracking a line trajectory of a square slider object with a single
#  contacter pusher.
#  -------------------------------------------------------------------

## Import Libraries
#  -------------------------------------------------------------------
import numpy as np
import numpy.matlib as nplib
from scipy.integrate import dblquad 
import casadi as cs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
#  -------------------------------------------------------------------

## Set Problem constants
#  -------------------------------------------------------------------
g = 9.81 # gravity acceleration constant in meter per second square
a = 0.09 # side dimension of the square slider in meters
m = 0.827 # mass of the slider in kilo grams
miu_g = 0.35 # coeficient of friction between slider and table
miu_p = 0.3 # coeficient of friction between pusher and slider
T = 5 # time of the simulation is seconds
freq = 50 # numer of increments per second
r_pusher = 0.01 # radious of the cilindrical pusher in meter
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
N = T*freq # total number of iterations
h = 1./freq # time interval of each iteration
A = a**2 # area of the slider in meter square
f_max = miu_g*m*g # limit force in Newton
# Area integral of norm of the distance for a square:
int_square = lambda a: dblquad(lambda x,y: np.square(x**2 + y**2), -a/2, a/2, -a/2, a/2)[0]
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
x = cs.SX.sym('x', 4)
# u - control vector
# u[0] - normal force in the local frame
# u[1] - tangential force in the local frame
# u[2] - relative sliding velocity between pusher and slider
u = cs.SX.sym('u', 3)
#  -------------------------------------------------------------------

## Build Motion Model
#  -------------------------------------------------------------------
L = cs.diag(cs.SX([f_max,f_max,m_max]))
ctheta = cs.cos(x[2]); stheta = cs.sin(x[2])
R = cs.SX(3,3)
R[0,0] = ctheta; R[0,1] = -stheta; R[1,0] = stheta; R[1,1] = ctheta; R[2,2] = 1;
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
B_func = cs.Function('B', [x], [B])
#  -------------------------------------------------------------------

## Generate Nominal Trajectory (line)
#  -------------------------------------------------------------------
# constant input and initial state
u0 = cs.SX(3,1); u0[0] = 0.03
x0 = cs.SX(4,1)
#  -------------------------------------------------------------------
t = cs.SX.sym('t'); ts = np.linspace(0, T, N)
dae = {'x':x, 't':t, 'ode': f_func(x, u0)}
F = cs.integrator('F', 'cvodes', dae, {'grid':ts, 'output_t0':True})
X_nom = np.array(F(x0=cs.DM(x0).full())['xf'])
U_nom = cs.repmat(u0, 1, N)
#  -------------------------------------------------------------------

## Set up QP Optimization Problem
#  -------------------------------------------------------------------
## ---- Input variables ---
X_bar = cs.SX.sym('x_bar', 4, N)
U_bar = cs.SX.sym('u_bar', 3, N)
## ---- Optimization objective ----------
Qcost = 10*cs.diag(cs.SX([3,3,0.1,0]))
QNcost = 2000*cs.diag(cs.SX([3,3,0.1,0]))
Rcost = 0.5*cs.diag(cs.SX([1,1,0.01]))
Cost = cs.dot(X_bar[:,-1],cs.mtimes(QNcost,X_bar[:,-1]))
for i in range(N-1):
    Cost += cs.dot(X_bar[:,i],cs.mtimes(Qcost,X_bar[:,i])) + cs.dot(U_bar[:,i],cs.mtimes(Rcost,U_bar[:,i]))
## ---- Initialize variables for optimization problem ---
w=[]
w0=[]
g = []
lbg = []
ubg = []
## ---- Initial Conditions ----
# ... 
# put here the starting state
## ---- Dynamic constraints ----
for i in range(N-1):
    Ai = A_func(X_nom[:,i], U_nom[:,i])
    Bi = B_func(X_nom[:,i])
    g += [X_bar[:,i+1]-h*(cs.mtimes(Ai,X_bar[:,i])-cs.mtimes(Bi,U_bar[:,i]))]
    lbg += [0, 0, 0, 0]
    ubg += [0, 0, 0, 0]
## ---- Control constraints ----
for i in range(N):
    # normal force
    g += [(U_bar[0,i]+U_nom[0,i])]
    lbg += [0]
    ubg += [cs.inf]
    # tangential force
    g += [miu_p*(U_bar[0,i]+U_nom[0,i])-cs.fabs(U_bar[1,i]+U_nom[1,i])]
    lbg += [0]
    ubg += [cs.inf]
    # relative sliding velocity
    g += [U_bar[2,i]+U_nom[2,i]]
    lbg += [0]
    ubg += [0]
## ---- Setting up optimization variables ---
for i in range(N):
    w += [X_bar[:,i]]
    w0 += [X_nom[:,i]]
    w += [U_bar[:,i]]
    w0 += [0, 0, 0]
## ---- Create solver ----
prob = {'f': Cost, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
solver = cs.nlpsol('solver', 'ipopt', prob);
## ---- Solve optimization problem ----
sol = solver(x0=cs.vertcat(*w0), lbg=lbg, ubg=ubg)
w_opt = sol['x']
X_bar_opt = cs.vertcat(w_opt[0::7].T,w_opt[1::7].T,w_opt[2::7].T,w_opt[3::7].T)
U_bar_opt = cs.vertcat(w_opt[4::7].T,w_opt[5::7].T,w_opt[6::7].T)
X_opt = X_bar_opt + X_nom
U_opt = U_bar_opt + U_nom
print(X_opt.shape)
print(U_opt.shape)
print(X_opt[:,0])
#  -------------------------------------------------------------------

# Animation of Nominal Trajectory
#  -------------------------------------------------------------------
## # set up the figure and subplot
## fig = plt.figure()
## fig.canvas.set_window_title('Matplotlib Animation')
## ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, \
##         xlim=(-0.1,0.5), ylim=(-0.1,0.5) \
## )
## ax.plot(X_nom[0,:], X_nom[1,:], color='red', linewidth=2.0, linestyle='dashed')
## ax.plot(X_nom[0,0], X_nom[1,0], X_nom[0,-1], X_nom[1,-1], marker='o', color='red')
## ax.grid(); ax.set_axisbelow(True)
## ax.set_title('Pusher-Slider Motion Animation')
## slider = patches.Rectangle(x0[0:2]-np.array([a/2, a/2]), a, a, x0[2])
## pusher = patches.Circle(np.array(p_pusher_func(x0)), radius=r_pusher, color='black')
## def init():
##     ax.add_patch(slider)
##     ax.add_patch(pusher)
##     return []
##     #return slider,
## def animate(i, slider, pusher):
##     slider.set_xy([X_nom[0,i]-a/2, X_nom[1,i]-a/2])
##     pusher.set_center(np.array(p_pusher_func(X_nom[:,i])))
##     return []
## # call the animation
## ani = animation.FuncAnimation(fig, animate, init_func=init, \
##         fargs=(slider,pusher,),
##         frames=N,
##         interval=T,
##         blit=True, repeat=False)
## ## to save animation, uncomment the line below:
## ## ani.save('sliding_nominal_traj.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
## #show the animation
## plt.show()
#  -------------------------------------------------------------------
