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
import casadi as cs
import sliding_pack

class MPC_nlpClass():

    def __init__(self, dyn_class, TH, Xnom, f_lim, psi_dot_lim, psi_lim, dt=0.1):

        # init parameters
        self.TH = TH
        self.dyn = dyn_class
        self.Xnom = Xnom
        self.f_lim = f_lim
        self.psi_dot_lim = psi_dot_lim
        self.psi_lim = psi_lim

        # initialize variables for opt and args
        self.opt = sliding_pack.opt.OptVars()
        self.opt.x = []
        self.args = sliding_pack.opt.OptArgs()
        self.args.x0 = []
        self.args.lbx = []
        self.args.ubx = []

        # set optimization variables
        self.X = cs.SX.sym('X', self.dyn.Nx, self.TH)
        self.U = cs.SX.sym('U', self.dyn.Nu, self.TH-1)
        # initial state
        self.x0 = cs.SX.sym('x0', self.dyn.Nx)
        self.del_cc = cs.SX.sym('del_cc', self.TH-1)

    def buildProblem(self):

        ## ---- Set optimization variables ----
        for i in range(self.TH-1):
            ## ---- Add States to optimization variables ---
            self.opt.x += self.X[:,i].elements()
            self.args.x0 += self.Xnom[:,i].elements()
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
        self.args.x0 += self.Xnom[:,-1].elements()
        self.args.lbx += [-cs.inf]*(self.dyn.Nx-1)
        self.args.ubx += [cs.inf]*(self.dyn.Nx-1)
        self.args.lbx += [-self.psi_lim]
        self.args.ubx += [self.psi_lim]

#    def solveProblem(self):
#        # 

#    def decodeSol(self):
#        #
