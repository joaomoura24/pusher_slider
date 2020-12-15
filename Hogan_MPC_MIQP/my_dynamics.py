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
import numpy as np
from scipy import integrate
import casadi as cs

## -------------------------------------------------------------------
## python lambda functions
## -------------------------------------------------------------------
int_square_np = lambda sq_side: integrate.dblquad(lambda x,y: np.sqrt((x**2) + (y**2)),
        -sq_side/2, sq_side/2, -sq_side/2, sq_side/2)[0]
int_quad_np = lambda sq_side: integrate.quad(lambda var: var**2, 
        -sq_side/2, sq_side/2)[0]
## -------------------------------------------------------------------
## casadi auxiliary functions
## -------------------------------------------------------------------
# Fixed step Runge-Kutta 4 integrator
M = 4 # RK4 steps per interval
N = 4 # number of control intervals
side_lenght = cs.SX.sym('side_lenght')
x = cs.SX.sym('x')
y = cs.SX.sym('y')
DX = side_lenght/(N*M)
DY = side_lenght/(N*M)
## -------------------------------------------------------------------
# cost function
g = cs.Function('g_ext', [x], [DX, (x**2)*DX])
Q = 0 # initialize cost
xx = -side_lenght/2 # initialize initial cond
for n in range(N):
    for m in range(M):
       k1, k1_q = g(xx)
       k2, k2_q = g(xx + k1/2)
       k3, k3_q = g(xx + k2/2)
       k4, k4_q = g(xx + k3)
       Q += (k1_q + 2*k2_q + 2*k3_q + k4_q)/6
       xx += (k1 +2*k2 +2*k3 +k4)/6
int_quad_cs = cs.Function('int_quad_cs', [side_lenght], [Q])
## -------------------------------------------------------------------
g = cs.Function('h_ext', [x, y], [DX, DY, (cs.sqrt((x**2)+(y**2)))*DX*DY])
Q = 0 # initialize cost
yy = -side_lenght/2 # initialize initial cond
for ny in range(N):
    for my in range(M):
        xx = -side_lenght/2
        for nx in range(N):
            for mx in range(M):
               k1_x, k1_y, k1_q = g(xx, yy)
               k2_x, k2_y, k2_q = g(xx + k1_x/2, yy + k1_y/2)
               k3_x, k3_y, k3_q = g(xx + k2_x/2, yy + k2_y/2)
               k4_x, k4_y, k4_q = g(xx + k3_x, yy + k3_y)
               Q += (k1_q + 2*k2_q + 2*k3_q + k4_q)/6
               xx += (k1_x +2*k2_x +2*k3_x +k4_x)/6
        yy += (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
int_square_cs = cs.Function('int_square_cs', [side_lenght], [Q])
side_const = 1.0
print('np: ', int_quad_np(side_const))
print('cs: ', int_quad_cs(side_const))
print('np: ', int_square_np(side_const))
print('cs: ', int_square_cs(side_const))
sys.exit(1)

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
A = cs.SX.sym('A') # slider area
int_A = cs.SX.sym('int_A') # integral of the norm of position vector
b = cs.veccat(A, int_A)
## -------------------------------------------------------------------
## Build Motion Model
c = int_A/A # ellipsoid approximation ratio
L = cs.SX.sym('L', cs.Sparsity.diag(3))
L[0,0] = L[1,1] = 1; L[2,2] = 1/(c**2);
ctheta = cs.cos(x[2]); stheta = cs.sin(x[2])
R = cs.SX(3,3)
R[0,0] = ctheta; R[0,1] = -stheta; R[1,0] = stheta; R[1,1] = ctheta; R[2,2] = 1;
R_func = cs.Function('R', [x], [R])


#  -------------------------------------------------------------------
# test
a = cs.SX.sym('a')
b1 = cs.SX.sym('b1')
b2 = cs.SX.sym('b2')
h = cs.Function('h', [b1,b2], [b1+b2])
f = cs.Function('f', [a,b1,b2], [a+h(b1,b2)])
print(cs.SX.is_constant(b1))
# print(cs.SX.is_constant(2.0))
print(f)
g = cs.Function('g', [a], [f(a, 2, 2)])
print(g)
print(f(a,2,2))
c1 = cs.SX.sym('c1')
c2 = cs.SX.sym('c2')
d = cs.SX.sym('d')
print(h(c1,c2))
print(h(1.0,1.0))
print(g(d))
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
# f = cs.SX(cs.vertcat(cs.mtimes(cs.mtimes(R,L),cs.mtimes(B,u[0:2])),u[2]))
# f_func = cs.Function('f', [x,u,b], [f])
# #  -------------------------------------------------------------------

