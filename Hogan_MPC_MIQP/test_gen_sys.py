## Import Libraries
#  -------------------------------------------------------------------
import numpy
import scipy.integrate
import casadi
#  -------------------------------------------------------------------

## Set Problem constants
#  -------------------------------------------------------------------
g = 9.81 # gravity acceleration constant in meters per second square
a = 0.09 # side dimension of the square slider in meters
m = 0.827 # mass of the slider
miu_g = 0.35 # coeficient of friction between slider and table
#  -------------------------------------------------------------------
## Computing Problem constants
#  -------------------------------------------------------------------
A = a**2 # area of the slider in meters
f_max = miu_g*m*g # limit force
# Area integral of norm of the distance for a square:
int_square = lambda a: scipy.integrate.dblquad(lambda x,y: numpy.square(x**2 + y**2), -a/2, a/2, -a/2, a/2)[0]
int_A = int_square(a)
m_max = miu_g*m*g*int_A/A # limit torque
#  -------------------------------------------------------------------

## Define states and actions
#  -------------------------------------------------------------------
# x - state vector
# x[0] - x slider CoM position in the global frame
# x[1] - y slider CoM position in the global frame
# x[2] - slider orientation in the global frame
# x[3] - angle of pusher relative to slider
x = casadi.SX.sym('x', 4)
# u - control vector
# u[0] - normal force in the local frame
# u[1] - tangential force in the local frame
# u[2] - relative sliding velocity between pusher and slider
u = casadi.SX.sym('u', 3)
#  -------------------------------------------------------------------

## Build Motion Model
#  -------------------------------------------------------------------
L = casadi.diag(casadi.SX([f_max,f_max,m_max]))
ctheta = casadi.cos(x[2]); stheta = casadi.sin(x[2])
R = casadi.SX(3,3)
R[0,0] = ctheta; R[0,1] = -stheta; R[1,0] = stheta; R[1,1] = ctheta; R[2,2] = 1;
xc = -a/2; yc = (a/2)*casadi.sin(x[3])
Jc = casadi.SX(2,3)
Jc[0,0] = 1; Jc[1,1] = 1; Jc[0,2] = -yc; Jc[1,2] = xc;
B = casadi.SX(Jc.T)
#  -------------------------------------------------------------------
f = casadi.SX(casadi.vertcat(casadi.mtimes(casadi.mtimes(R,L),casadi.mtimes(B,u[0:2])),u[2]))
f_func = casadi.Function('f', [x,u], [f])
#  -------------------------------------------------------------------
