## Author: Joao Moura
## Contact: jpousad@ed.ac.uk
## Date: 19/10/2020
## -------------------------------------------------------------------
## Description:
## 
## Functions modelling the dynamics of an object sliding on a table.
## Based on: Hogan F.R, Rodriguez A. (2020) IJRR paper
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## Import libraries
## -------------------------------------------------------------------
import sys
import casadi as cs
import my_integration

## -------------------------------------------------------------------
## build dynamic fuction for quasi-static ellipsoidal limit surface
## -------------------------------------------------------------------
## x - state vector
x_slider = cs.SX.sym('x_slider') # in global frame [m]
y_slider = cs.SX.sym('y_slider') # in global frame [m]
theta = cs.SX.sym('theta') # in global frame [rad]
psi = cs.SX.sym('psi') # in relative frame [rad]
x = cs.veccat(x_slider, y_slider, theta, psi)
## u - control vector
v_norm = cs.SX.sym('v_norm') # in local frame [m/s]
v_tan = cs.SX.sym('v_tan') # in  local frame [m/s]
psi_dot = cs.SX.sym('psi_dot') # rel vel between pusher and slider [rad/s]
u = cs.veccat(v_norm, v_tan, psi_dot)
# beta - dynamic parameters
sl = cs.SX.sym('sl') # slider side lenght
r_pusher = cs.SX.sym('r_pusher') # radious of the cilindrical pusher
beta = cs.veccat(sl, r_pusher)
## -------------------------------------------------------------------
## Build Motion Model
## -------------------------------------------------------------------
## Rotation matrix
Area = sl**2
int_Area = my_integration.int_square_cs(sl)
c = int_Area/Area # ellipsoid approximation ratio
A = cs.SX.sym('A', cs.Sparsity.diag(3))
A[0,0] = A[1,1] = 1; A[2,2] = 1/(c**2);
ctheta = cs.cos(theta); stheta = cs.sin(theta)
R = cs.SX(3,3)
R[0,0] = ctheta; R[0,1] = -stheta; R[1,0] = stheta; R[1,1] = ctheta; R[2,2] = 1.0;
square_slider_quasi_static_ellipsoidal_limit_surface_R = cs.Function('square_slider_quasi_static_ellipsoidal_limit_surface_R', [x], [R])
#  -------------------------------------------------------------------
## slider position
xc = -sl/2; yc = (sl/2)*cs.sin(psi)
rc = cs.SX(2,1); rc[0] = xc-r_pusher; rc[1] = yc
p_pusher = cs.mtimes(R[0:2,0:2], rc)[0:2] + x[0:2]
square_slider_quasi_static_ellipsoidal_limit_surface_p = cs.Function('square_slider_quasi_static_ellipsoidal_limit_surface_p', [x,beta], [p_pusher])
#  -------------------------------------------------------------------
## dynamics
Jc = cs.SX(2,3)
Jc[0,0] = 1; Jc[1,1] = 1; Jc[0,2] = -yc; Jc[1,2] = xc;
# B = cs.SX(Jc.T)
f = cs.SX(cs.vertcat(cs.mtimes(cs.mtimes(R,A),cs.mtimes(Jc.T,u[0:2])),u[2]))
# square_slider_quasi_static_ellipsoid_fric = cs.Function('f', [x,u,beta], [f])
square_slider_quasi_static_ellipsoidal_limit_surface_f = cs.Function('square_slider_quasi_static_ellipsoidal_limit_surface_f', [x,u,beta], [f])
#  -------------------------------------------------------------------

# #  -------------------------------------------------------------------
# xc = -a/2; yc = (a/2)*cs.sin(x[3])
# Jc = cs.SX(2,3)
# Jc[0,0] = 1; Jc[1,1] = 1; Jc[0,2] = -yc; Jc[1,2] = xc;
# B = cs.SX(Jc.T)
# #  -------------------------------------------------------------------
# rc = cs.SX(2,1); rc[0] = xc-r_pusher; rc[1] = yc
# p_pusher = cs.mtimes(R[0:2,0:2], rc)[0:2] + x[0:2]
# p_pusher_func = cs.Function('p_pusher', [x], [p_pusher])
# #  -------------------------------------------------------------------
# f = cs.SX(cs.vertcat(cs.mtimes(cs.mtimes(R,A),cs.mtimes(B,u[0:2])),u[2]))
# f_func = cs.Function('f', [x,u,b], [f])
# #  -------------------------------------------------------------------

