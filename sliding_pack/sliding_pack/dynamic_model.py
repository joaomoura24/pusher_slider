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
import sys
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import casadi as cs
import sliding_pack

class Sys_sq_slider_quasi_static_ellip_lim_surf():

    def __init__(self, configDict, contactMode='sticking'):

        # init parameters
        self.mode = contactMode
        # self.sl = configDict['sideLenght']  # side dimension of the square slider [m]
        self.xl = configDict['xLenght']  # x dim of the slider [m]
        self.yl = configDict['yLenght']  # y dim of the slider [m]
        self.r_pusher = configDict['pusherRadious']  # radius of the pusher [m]
        self.miu = configDict['pusherFricCoef']  # fric between pusher and slider
        self.f_lim = configDict['pusherForceLim']
        self.psi_dot_lim = configDict['pusherAngleVelLim']
        self.psi_lim = 0.95*np.arctan2(self.xl, self.yl)
        self.Kz_max = configDict['Kz_max']
        self.Kz_min = configDict['Kz_min']
        # vector of physical parameters
        self.beta = [self.xl, self.yl, self.r_pusher]

        # system constant variables
        self.Nx = 4  # number of state variables

        # vectors of state and control
        #  -------------------------------------------------------------------
        # x - state vector
        # x[0] - x slider CoM position in the global frame
        # x[1] - y slider CoM position in the global frame
        # x[2] - slider orientation in the global frame
        # x[3] - angle of pusher relative to slider
        self.x = cs.SX.sym('x', self.Nx)
        # dx - derivative of the state vector
        self.dx = cs.SX.sym('dx', self.Nx)
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
        __f_tan = cs.SX.sym('__f_tan')  # in local frame [N]
        # rel vel between pusher and slider [rad/s]
        __psi_dot = cs.SX.sym('__psi_dot')
        __u = cs.veccat(__f_norm, __f_tan, __psi_dot)
        # beta - dynamic parameters
        __xl = cs.SX.sym('__xl')  # slider x lenght
        __yl = cs.SX.sym('__yl')  # slider y lenght
        __r_pusher = cs.SX.sym('__r_pusher')  # radious of the cilindrical pusher
        __beta = cs.veccat(__xl, __yl, __r_pusher)

        # system model
        # -------------------------------------------------------------------
        # Rotation matrix
        __Area = __xl*__yl
        __int_Area = sliding_pack.integral.rect_cs(__xl, __yl)
        __c = __int_Area/__Area # ellipsoid approximation ratio
        __A = cs.SX.sym('__A', cs.Sparsity.diag(3))
        __A[0,0] = __A[1,1] = 1.; __A[2,2] = 1./(__c**2);
        __ctheta = cs.cos(__theta)
        __stheta = cs.sin(__theta)
        __R = cs.SX(3, 3)
        __R[0,0] = __ctheta; __R[0,1] = -__stheta; __R[1,0] = __stheta; __R[1,1] = __ctheta; __R[2,2] = 1.0;
        #  -------------------------------------------------------------------
        self.R = cs.Function('R', [__x], [__R], ['x'], ['R'])
        #  -------------------------------------------------------------------
        # slider position
        __xc = -__xl/2; __yc = -(__yl/2)*cs.tan(__psi)
        __rc = cs.SX(2,1); __rc[0] = __xc-__r_pusher; __rc[1] = __yc
        __p_pusher = cs.mtimes(__R[0:2,0:2], __rc)[0:2] + __x[0:2]
        #  -------------------------------------------------------------------
        __p = cs.SX.sym('p', 2) # slider position
        __rc_prov = cs.mtimes(__R[0:2,0:2].T, __p - __x[0:2])
        __psi_prov = -cs.atan2(__rc_prov[1], __xl/2)
        self.psi_ = cs.Function('psi_', [__x,__p,__beta], [__psi_prov])
        self.psi = cs.Function('psi', [self.x,__p], [self.psi_(self.x, __p, self.beta)])
        #  -------------------------------------------------------------------
        self.p_ = cs.Function('p_', [__x,__beta], [__p_pusher], ['x', 'b'], ['p'])
        self.p = cs.Function('p', [self.x], [self.p_(self.x, self.beta)], ['x'], ['p'])
        #  -------------------------------------------------------------------
        self.s = cs.Function('s', [self.x], [self.x[0:3]], ['x'], ['s'])
        #  -------------------------------------------------------------------
        # dynamics
        __Jc = cs.SX(2,3)
        __Jc[0,0] = 1; __Jc[1,1] = 1; __Jc[0,2] = -__yc; __Jc[1,2] = __xc;
        __f = cs.SX(cs.vertcat(cs.mtimes(cs.mtimes(__R,__A),cs.mtimes(__Jc.T,__u[0:2])),__u[2]))
        #  -------------------------------------------------------------------
        self.f_ = cs.Function('f_', [__x,__u,__beta], [__f], ['x', 'u', 'b'], ['f'])
        #  -------------------------------------------------------------------

        # control constraints
        #  -------------------------------------------------------------------
        if self.mode == 'sliding_cc':
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider counterclockwise
            # u[3] - rel sliding vel between pusher and slider clockwise
            self.Nu = 4  # number of action variables
            self.u = cs.SX.sym('u', self.Nu)
            self.Nz = 0
            self.z0 = []
            self.lbz = []
            self.ubz = []
            # discrete extra variable
            self.z_discrete = False
            empty_var = cs.SX.sym('empty_var')
            self.g_u = cs.Function('g_u', [self.u, empty_var], [cs.vertcat(
                # friction cone edges
                self.miu*self.u[0]+self.u[1],
                self.miu*self.u[0]-self.u[1],
                # complementarity constraint
                (self.miu * self.u[0] - self.u[1])*self.u[3],
                (self.miu * self.u[0] + self.u[1])*self.u[2]
            )], ['u', 'other'], ['g'])
            self.g_lb = [0., 0., 0., 0.]
            self.g_ub = [cs.inf, cs.inf, 0., 0.]
            self.Ng_u = 4
            # cost gain for extra variable
            __Ks_max = self.Kz_max
            __Ks_min = self.Kz_min
            __i_th = cs.SX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [__Ks_max * cs.exp(__i_th * cs.log(__Ks_min / __Ks_max))])
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim, 0.0, 0.0]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim, self.psi_dot_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u], [self.f_(self.x, cs.vertcat(self.u[0:2], self.u[2]-self.u[3]), self.beta)],  ['x', 'u'], ['f'])
        elif self.mode == 'sliding_cc_slack':
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider counterclockwise
            # u[3] - rel sliding vel between pusher and slider clockwise
            self.Nu = 4  # number of action variables
            self.u = cs.SX.sym('u', self.Nu)
            self.Nz = 2
            self.z = cs.SX.sym('z', self.Nz)
            self.z0 = [1.]*self.Nz
            self.lbz = [-cs.inf]*self.Nz
            self.ubz = [cs.inf]*self.Nz
            # discrete extra variable
            self.z_discrete = False
            self.g_u = cs.Function('g_u', [self.u, self.z], [cs.vertcat(
                # friction cone edges
                self.miu*self.u[0]+self.u[1],
                self.miu*self.u[0]-self.u[1],
                # complementarity constraint
                (self.miu * self.u[0] - self.u[1])*self.u[3] + self.z[0],
                (self.miu * self.u[0] + self.u[1])*self.u[2] + self.z[1]
            )], ['u', 'other'], ['g'])
            self.g_lb = [0., 0., 0., 0.]
            self.g_ub = [cs.inf, cs.inf, 0., 0.]
            self.Ng_u = 4
            # cost gain for extra variable
            __Ks_max = self.Kz_max
            __Ks_min = self.Kz_min
            __i_th = cs.SX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [__Ks_max * cs.exp(__i_th * cs.log(__Ks_min / __Ks_max))])
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim, 0.0, 0.0]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim, self.psi_dot_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u], [self.f_(self.x, cs.vertcat(self.u[0:2], self.u[2]-self.u[3]), self.beta)],  ['x', 'u'], ['f'])
        elif self.mode == 'sliding_mi':
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            # u[2] - rel sliding vel between pusher and slider
            self.Nu = 3  # number of action variables
            self.u = cs.SX.sym('u', self.Nu)
            self.Nz = 3
            self.z = cs.SX.sym('z', self.Nz)
            self.z0 = [0]*self.Nz
            self.lbz = [0]*self.Nz
            self.ubz = [1]*self.Nz
            # discrete extra variable
            self.z_discrete = True
            self.Ng_u = 7
            bigM = 500  # big M for the Mixed Integer optimization
            self.g_u = cs.Function('g_u', [self.u, self.z], [cs.vertcat(
                self.miu*self.u[0]+self.u[1] + bigM*self.z[1],  # friction cone edge
                self.miu*self.u[0]-self.u[1] + bigM*self.z[2],  # friction cone edge
                self.miu*self.u[0]+self.u[1] - bigM*(1-self.z[2]),  # friction cone edge
                self.miu*self.u[0]-self.u[1] - bigM*(1-self.z[1]),  # friction cone edge
                self.u[2] + bigM*self.z[2],  # relative rot constraint
                self.u[2] - bigM*self.z[1],
                cs.sum1(self.z),  # sum of the integer variables
            )], ['u', 'other'], ['g'])
            self.g_lb = [0., 0., -cs.inf, -cs.inf, 0., -cs.inf, 1.]
            self.g_ub = [cs.inf, cs.inf, 0., 0., cs.inf, 0., 1.]
            __i_th = cs.SX.sym('__i_th')
            self.kz_f = cs.Function('ks', [__i_th], [0.])
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.ubx = [cs.inf, cs.inf, cs.inf, self.psi_lim]
            self.lbu = [0.0,  -self.f_lim, 0.0]
            self.ubu = [self.f_lim, self.f_lim, self.psi_dot_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u], [self.f_(self.x, self.u, self.beta)],  ['x', 'u'], ['f'])
        elif self.mode == 'sticking':
            # u - control vector
            # u[0] - normal force in the local frame
            # u[1] - tangential force in the local frame
            self.Nu = 2  # number of action variables
            self.u = cs.SX.sym('u', self.Nu)
            empty_var = cs.SX.sym('empty_var')
            self.g_u = cs.Function('g_u', [self.u, empty_var], [cs.vertcat(
                self.miu*self.u[0]+self.u[1],  # friction cone edge
                self.miu*self.u[0]-self.u[1]  # friction cone edge
            )], ['u', 'other'], ['g'])
            self.g_lb = [0.0, 0.0]
            self.g_ub = [cs.inf, cs.inf]
            self.Nz = 0
            self.z0 = []
            self.lbz = []
            self.ubz = []
            # discrete extra variable
            self.z_discrete = False
            self.Ng_u = 2
            # state and acton limits
            #  -------------------------------------------------------------------
            self.lbx = [-cs.inf, -cs.inf, -cs.inf, 0.0]
            self.ubx = [cs.inf, cs.inf, cs.inf, 0.0]
            self.lbu = [0.0,  -self.f_lim]
            self.ubu = [self.f_lim, self.f_lim]
            #  -------------------------------------------------------------------
            # dynamics equation
            self.f = cs.Function('f', [self.x, self.u], [self.f_(self.x, cs.vertcat(self.u, 0.0), self.beta)],  ['x', 'u'], ['f'])
        else:
            print('Specified mode ``{}`` does not exist!'.format(self.mode))
            sys.exit(-1)
        #  -------------------------------------------------------------------

    def set_patches(self, ax, x_data):
        x0 = x_data[:, 0]
        d0 = np.array(cs.mtimes(self.R(x0), [-self.xl/2, -self.yl/2, 0]).T)[0]
        self.slider = patches.Rectangle(
                x0[0:2]+d0[0:2], self.xl, self.yl, x0[2])
        self.pusher = patches.Circle(
                np.array(self.p(x0)), radius=self.r_pusher, color='black')
        self.path_past, = ax.plot(x0[0], x0[1], color='orange')
        self.path_future, = ax.plot(x0[0], x0[1],
                color='orange', linestyle='dashed')
        ax.add_patch(self.slider)
        ax.add_patch(self.pusher)
        self.path_past.set_linewidth(2)

    def animate(self, i, ax, x_data, X_future=None):
        xi = x_data[:, i]
        # distance between centre of square reference corner
        di = np.array(cs.mtimes(self.R(xi), [-self.xl/2, -self.yl/2, 0]).T)[0]
        # square reference corner
        ci = xi[0:3] + di
        # compute transformation with respect to rotation angle xi[2]
        trans_ax = ax.transData
        coords = trans_ax.transform(ci[0:2])
        trans_i = transforms.Affine2D().rotate_around(
                coords[0], coords[1], xi[2])
        # Set changes
        self.slider.set_transform(trans_ax+trans_i)
        self.slider.set_xy([ci[0], ci[1]])
        self.pusher.set_center(np.array(self.p(xi)))
        # Set path changes
        if self.path_past is not None:
            self.path_past.set_data(x_data[0, 0:i], x_data[1, 0:i])
        if (self.path_future is not None) and (X_future is not None):
            self.path_future.set_data(X_future[0, :, i], X_future[1, :, i])
        return []
    #  -------------------------------------------------------------------
