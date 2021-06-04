# Author: Joao Moura
# Contact: jpousad@ed.ac.uk
# Date: 02/06/2021
# -------------------------------------------------------------------
# Description:
# 
# Class for the trajectory optimization (TO) for the pusher-slider 
# problem using a Non-Linear Program (NLP) approach
# -------------------------------------------------------------------

# import libraries
import numpy as np
import casadi as cs
import sliding_pack

class MPC_nlpClass():

    def __init__(self, dyn_class, TH, X_nom_val, f_lim, psi_dot_lim, psi_lim, dt=0.1):

        # init parameters
        self.TH = TH
        self.dyn = dyn_class
        self.X_nom_val = X_nom_val
        self.f_lim = f_lim
        self.psi_dot_lim = psi_dot_lim
        self.psi_lim = psi_lim

        # initialize variables for opt and args
        self.opt = sliding_pack.opt.OptVars()
        self.opt.x = []
        self.opt.g = []
        self.opt.f = []
        self.opt.p = []
        self.args = sliding_pack.opt.OptArgs()
        self.args.x0 = []
        self.args.lbx = []
        self.args.ubx = []
        self.args.lbg = []
        self.args.ubg = []

        # set optimization variables
        self.X = cs.SX.sym('X', self.dyn.Nx, self.TH)
        self.U = cs.SX.sym('U', self.dyn.Nu, self.TH-1)
        self.X_nom = cs.SX.sym('X_nom', self.dyn.Nx, TH)
        # initial state
        self.x0 = cs.SX.sym('x0', self.dyn.Nx)
        self.del_cc = cs.SX.sym('del_cc', self.TH-1)

        # constraint functions
        #  -------------------------------------------------------------------
        # ---- Define Dynamic constraints ----
        __x_next = cs.SX.sym('__x_next', self.dyn.Nx)
        f_error = cs.Function('f_error', [self.dyn.x, self.dyn.u, __x_next],
                [__x_next-self.dyn.x-dt*self.dyn.f(self.dyn.x,self.dyn.u)])
        self.F_error = f_error.map(TH-1)
        #  -------------------------------------------------------------------
        # control constraints
        self.fric_cone_c = cs.Function('fric_cone_c', [self.dyn.u], [cs.vertcat(
            self.dyn.miu*self.dyn.u[0]+self.dyn.u[1],
            self.dyn.miu*self.dyn.u[0]-self.dyn.u[1]
        )])
        self.fric_cone_C = self.fric_cone_c.map(self.TH-1)
        #  -------------------------------------------------------------------
        slack_var = cs.SX.sym('slack_var')
        complem_c = cs.Function('fric_cone_lim_c', [self.dyn.u, slack_var], [cs.vertcat(
            (self.dyn.miu * self.dyn.u[0] - self.dyn.u[1])*self.dyn.u[3] + slack_var +
            (self.dyn.miu * self.dyn.u[0] + self.dyn.u[1])*self.dyn.u[2]
        )])
        self.complem_C = complem_c.map(self.TH-1)
        #  -------------------------------------------------------------------

        #  -------------------------------------------------------------------
        x_nom = cs.SX.sym('x_nom', self.dyn.Nx)
        pos_err = self.dyn.x - x_nom
        W_f = cs.diag(cs.SX([1.0,1.0,0.01,0.0]))
        cost_f = cs.Function('cost_f', [self.dyn.x, x_nom], [cs.dot(pos_err,cs.mtimes(W_f,pos_err))])
        self.cost_F = cost_f.map(self.TH)
        Ks_max = 50.0; Ks_min = 0.1; xs = np.linspace(0,1,self.TH-1)
        self.Ks = Ks_max*cs.exp(xs*cs.log(Ks_min/Ks_max))
        #  -------------------------------------------------------------------

    def buildProblem(self):

        ## ---- Set optimization variables ----
        for i in range(self.TH-1):
            ## ---- Add States to optimization variables ---
            self.opt.x += self.X[:,i].elements()
            self.args.x0 += self.X_nom_val[:,i].elements()
            self.args.lbx += [-cs.inf, -cs.inf, -cs.inf, -self.psi_lim]
            self.args.ubx += [cs.inf, cs.inf, cs.inf, self.psi_lim]
            ## ---- Add Actions to optimization variables ---
            self.opt.x += self.U[:,i].elements()
            self.args.lbx += [0.0,  -self.f_lim,         0.0,         0.0]
            self.args.ubx += [self.f_lim, self.f_lim, self.psi_dot_lim, self.psi_dot_lim]
            self.args.x0 += [0.0,     0.0, 0.0, 0.0]
            ## ---- Add slack variables ---
            self.opt.x += self.del_cc[i].elements()
            self.args.x0 += [0.0]
            self.args.lbx += [-cs.inf]
            self.args.ubx += [cs.inf]
        self.opt.x += self.X[:,-1].elements()
        self.args.x0 += self.X_nom_val[:,-1].elements()
        self.args.lbx += [-cs.inf]*(self.dyn.Nx-1)
        self.args.ubx += [cs.inf]*(self.dyn.Nx-1)
        self.args.lbx += [-self.psi_lim]
        self.args.ubx += [self.psi_lim]

        ## ---- Set optimzation constraints ----
        self.opt.g = (self.X[:,0]-self.x0).elements() ## Initial Conditions
        self.args.lbg = [0.0]*self.dyn.Nx
        self.args.ubg = [0.0]*self.dyn.Nx
        # ---- Dynamic constraints ---- 
        self.opt.g += self.F_error(self.X[:, :-1], self.U, self.X[:, 1:]).elements()
        self.args.lbg += [0.] * self.dyn.Nx * (self.TH-1)
        self.args.ubg += [0.] * self.dyn.Nx * (self.TH-1)
        # ---- Friction cone constraints ----
        self.opt.g += self.fric_cone_C(self.U).elements()
        self.args.lbg += [0.0, 0.0] * (self.TH-1)
        self.args.ubg += [cs.inf, cs.inf] * (self.TH-1)
        # Complementary constraint
        self.opt.g += self.complem_C(self.U, self.del_cc.T).elements()
        self.args.lbg += [0.0] * (self.TH-1)
        self.args.ubg += [0.0] * (self.TH-1)

        ## ---- Set optimization parameters ----
        self.opt.p = []
        self.opt.p += self.x0.elements()
        self.opt.p += self.X_nom.elements()

        ## ---- optimization cost ----
        self.opt.f = cs.sum2(self.cost_F(self.X, self.X_nom))
        self.opt.f += cs.sum1(self.Ks*(self.del_cc**2))
#    def solveProblem(self):
#        # 

#    def decodeSol(self):
#        #
