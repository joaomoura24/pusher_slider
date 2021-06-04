# Author: Joao Moura
# Contact: jpousad@ed.ac.uk
# Date: 19/10/2020
# -------------------------------------------------------------------
# Description:
# 
# Functions modelling the dynamics of an object sliding on a table.
# Based on: Hogan F.R, Rodriguez A. (2020) IJRR paper
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------
import casadi as cs
import sliding_pack

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
int_Area = sliding_pack.integral.square_cs(sl)
c = int_Area/Area # ellipsoid approximation ratio
A = cs.SX.sym('A', cs.Sparsity.diag(3))
A[0,0] = A[1,1] = 1; A[2,2] = 1/(c**2);
ctheta = cs.cos(theta); stheta = cs.sin(theta)
R = cs.SX(3,3)
R[0,0] = ctheta; R[0,1] = -stheta; R[1,0] = stheta; R[1,1] = ctheta; R[2,2] = 1.0;
square_slider_quasi_static_ellipsoidal_limit_surface_R = cs.Function('square_slider_quasi_static_ellipsoidal_limit_surface_R', [x], [R])
#  -------------------------------------------------------------------
## slider position
xc = -sl/2; yc = -(sl/2)*cs.tan(psi)
rc = cs.SX(2,1); rc[0] = xc-r_pusher; rc[1] = yc
p_pusher = cs.mtimes(R[0:2,0:2], rc)[0:2] + x[0:2]
square_slider_quasi_static_ellipsoidal_limit_surface_p = cs.Function('square_slider_quasi_static_ellipsoidal_limit_surface_p', [x,beta], [p_pusher])
#  -------------------------------------------------------------------
## dynamics
Jc = cs.SX(2,3)
Jc[0,0] = 1; Jc[1,1] = 1; Jc[0,2] = -yc; Jc[1,2] = xc;
f = cs.SX(cs.vertcat(cs.mtimes(cs.mtimes(R,A),cs.mtimes(Jc.T,u[0:2])),u[2]))
square_slider_quasi_static_ellipsoidal_limit_surface_f = cs.Function('square_slider_quasi_static_ellipsoidal_limit_surface_f', [x,u,beta], [f])
#  -------------------------------------------------------------------

class System_square_slider_quasi_static_ellipsoidal_limit_surface():

    def __init__(self, slider_dim=0.09, pusher_radious=0.01, miu=0.3):

        # system constant variables
        self.Nx = 4  # number of state variables
        self.Nu = 4  # number of action variables

        # physical constant quantaties
        self.sl = slider_dim  # side dimension of the square slider [m]
        self.r_pusher = pusher_radious  # radius of the cylindrical pusher [m]
        self.miu = miu  # friction between pusher and slider
        # vector of physical parameters
        self.beta = [self.sl, self.r_pusher]

        # vectors of state and control
        #  -------------------------------------------------------------------
        # x - state vector
        # x[0] - x slider CoM position in the global frame
        # x[1] - y slider CoM position in the global frame
        # x[2] - slider orientation in the global frame
        # x[3] - angle of pusher relative to slider
        self.x = cs.SX.sym('x', self.Nx)
        # limits
        # self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
        # self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
        # dx - derivative of the state vector
        self.dx = cs.SX.sym('dx', self.Nx)
        # u - control vector
        # u[0] - normal force in the local frame
        # u[1] - tangential force in the local frame
        # u[2] - rel sliding vel between pusher and slider counterclockwise
        # u[3] - rel sliding vel between pusher and slider clockwise
        self.u = cs.SX.sym('u', self.Nu)
        #  -------------------------------------------------------------------

        # auxiliar symbolic variables
        # -------------------------------------------------------------------
        # x - state vector
        __x_slider = cs.SX.sym('__x_slider')  # in global frame [m]
        __y_slider = cs.SX.sym('__y_slider')  # in global frame [m]
        __theta = cs.SX.sym('__theta')  # in global frame [rad]
        __psi = cs.SX.sym('__psi')  # in relative frame [rad]
        __x = cs.veccat(__x_slider, __y_slider, __theta, __psi)
        # u - control vector
        __f_norm = cs.SX.sym('__f_norm')  # in local frame [N]
        __f_norm = cs.SX.sym('__f_norm')  # in  local frame [N]
        # rel vel between pusher and slider [rad/s]
        __psi_dot_ccw = cs.SX.sym('__psi_dot_ccw')
        # rel vel between pusher and slider [rad/s]
        __psi_dot_cw = cs.SX.sym('__psi_dot_cw')
        __u = cs.veccat(v_norm, __f_norm, __psi_dot_ccw, __psi_dot_cw)
        # beta - dynamic parameters
        __sl = cs.SX.sym('__sl')  # slider side lenght
        __r_pusher = cs.SX.sym('__r_pusher')  # radious of the cilindrical pusher
        __beta = cs.veccat(__sl, __r_pusher)

        # system model
        # -------------------------------------------------------------------
        # Rotation matrix
        __Area = __sl**2
        __int_Area = sliding_pack.integral.square_cs(__sl)
        __c = __int_Area/__Area # ellipsoid approximation ratio
        __A = cs.SX.sym('__A', cs.Sparsity.diag(3))
        __A[0,0] = __A[1,1] = 1; __A[2,2] = 1/(__c**2);
        __ctheta = cs.cos(__theta)
        __stheta = cs.sin(__theta)
        __R = cs.SX(3, 3)
        __R[0,0] = __ctheta; __R[0,1] = -__stheta; __R[1,0] = __stheta; __R[1,1] = __ctheta; __R[2,2] = 1.0;
        #  -------------------------------------------------------------------
        self.R = cs.Function('R', [__x], [__R], ['x'], ['R'])
        #  -------------------------------------------------------------------
        # slider position
        __xc = -__sl/2; __yc = -(__sl/2)*cs.tan(__psi)
        __rc = cs.SX(2,1); __rc[0] = __xc-__r_pusher; __rc[1] = __yc
        __p_pusher = cs.mtimes(__R[0:2,0:2], __rc)[0:2] + __x[0:2]
        #  -------------------------------------------------------------------
        self.p_ = cs.Function('p_', [__x,__beta], [__p_pusher], ['x', 'b'], ['p'])
        self.p = cs.Function('p', [self.x], [self.p_(self.x, self.beta)], ['x'], ['p'])
        #  -------------------------------------------------------------------
        # dynamics
        __Jc = cs.SX(2,3)
        __Jc[0,0] = 1; __Jc[1,1] = 1; __Jc[0,2] = -__yc; __Jc[1,2] = __xc;
        __f = cs.SX(cs.vertcat(cs.mtimes(cs.mtimes(__R,__A),cs.mtimes(__Jc.T,__u[0:2])),__u[2]-__u[3]))
        #  -------------------------------------------------------------------
        self.f_ = cs.Function('f_', [__x,__u,__beta], [__f], ['x', 'u', 'b'], ['f'])
        self.f = cs.Function('f', [self.x, self.u], [self.f_(self.x, self.u, self.beta)],  ['x', 'u'], ['f'])
        #  -------------------------------------------------------------------

