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
import sys
import os
import time
import numpy as np
import casadi as cs
import sliding_pack

class MPC_nlpClass():

    def __init__(self, dyn_class, TH, X_nom_val, dt=0.1):

        # init parameters
        self.TH = TH
        self.dyn = dyn_class
        self.X_nom_val = X_nom_val

        # opt var dimensionality
        self.Nxu = self.dyn.Nx + self.dyn.Nu
        N_s = 1
        self.Nopt = self.Nxu + N_s

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
        self.f_error = cs.Function('f_error', [self.dyn.x, self.dyn.u, __x_next],
                [__x_next-self.dyn.x-dt*self.dyn.f(self.dyn.x,self.dyn.u)])
        self.F_error = self.f_error.map(TH-1)
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
        self.G_u = self.dyn.g_u.map(self.TH-1)
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
        self.G_u = self.dyn.g_u.map(self.TH-1)
        #  -------------------------------------------------------------------)
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

    def buildProblem(self, solver_name, code_gen=False, no_printing=False):

        # ---- Set optimization variables ----
        for i in range(self.TH-1):
            # ---- Add States to optimization variables ---
            self.opt.x += self.X[:, i].elements()
            self.args.lbx += self.dyn.lbx
            self.args.ubx += self.dyn.ubx
            self.args.x0 += self.X_nom_val[:, i].elements()
            # ---- Add Actions to optimization variables ---
            self.opt.x += self.U[:, i].elements()
            self.args.lbx += self.dyn.lbu
            self.args.ubx += self.dyn.ubu
            self.args.x0 += [0.0]*self.dyn.Nu
            # ---- Add slack variables ---
            self.opt.x += self.del_cc[i].elements()
            self.args.x0 += [0.0]
            self.args.lbx += [-cs.inf]
            self.args.ubx += [cs.inf]
        self.opt.x += self.X[:, -1].elements()
        self.args.lbx += self.dyn.lbx
        self.args.ubx += self.dyn.ubx
        self.args.x0 += self.X_nom_val[:, -1].elements()

        # ---- Set optimzation constraints ----
        self.opt.g = (self.X[:,0]-self.x0).elements()  # Initial Conditions
        self.args.lbg = [0.0]*self.dyn.Nx
        self.args.ubg = [0.0]*self.dyn.Nx
        # for i in range(self.TH-1):
        #     self.opt.g += self.f_error(self.X[:, i], self.U[:, i], self.X[:, i+1]).elements()
        #     self.args.lbg += [0.] * self.dyn.Nx
        #     self.args.ubg += [0.] * self.dyn.Nx
        #     self.opt.g += self.dyn.g_u(self.U[:, i], self.del_cc[i]).elements()
        #     self.args.lbg += self.dyn.g_lb
        #     self.args.ubg += self.dyn.g_ub
        # ---- Dynamic constraints ---- 
        self.opt.g += self.F_error(self.X[:, :-1], self.U, self.X[:, 1:]).elements()
        self.args.lbg += [0.] * self.dyn.Nx * (self.TH-1)
        self.args.ubg += [0.] * self.dyn.Nx * (self.TH-1)
        # ---- Friction constraints ----
        self.opt.g += self.G_u(self.U, self.del_cc.T).elements()
        self.args.lbg += self.dyn.g_lb * (self.TH-1)
        self.args.ubg += self.dyn.g_ub * (self.TH-1)
        # # ---- Friction cone constraints ----
        # self.opt.g += self.fric_cone_C(self.U).elements()
        # self.args.lbg += [0.0, 0.0] * (self.TH-1)
        # self.args.ubg += [cs.inf, cs.inf] * (self.TH-1)
        # # Complementary constraint
        # self.opt.g += self.complem_C(self.U, self.del_cc.T).elements()
        # self.args.lbg += [0.0] * (self.TH-1)
        # self.args.ubg += [0.0] * (self.TH-1)

        ## ---- Set optimization parameters ----
        self.opt.p = []
        self.opt.p += self.x0.elements()
        self.opt.p += self.X_nom.elements()

        ## ---- optimization cost ----
        self.opt.f = cs.sum2(self.cost_F(self.X, self.X_nom))
        self.opt.f += cs.sum1(self.Ks*(self.del_cc**2))

        # Set up QP Optimization Problem
        #  -------------------------------------------------------------------
        # ---- Set solver options ----
        opts_dict = {'print_time': 0}
        prog_name = 'MPC' + '_TH' + str(self.TH) + '_' + solver_name + '_codeGen_' + str(code_gen)
        if solver_name == 'ipopt':
            if no_printing: opts_dict['ipopt.print_level'] = 0
            opts_dict['ipopt.jac_d_constant'] = 'yes'
            opts_dict['ipopt.warm_start_init_point'] = 'yes'
            opts_dict['ipopt.hessian_constant'] = 'yes'
        if solver_name == 'snopt':
            if no_printing: opts_dict['snopt'] = {'Major print level': '0', 'Minor print level': '0'}
            opts_dict['snopt']['Hessian updates'] = 1
        if solver_name == 'qpoases':
            if no_printing: opts_dict['printLevel'] = 'none'
            opts_dict['sparse'] = True
        if solver_name == 'gurobi':
            if no_printing: opts_dict['gurobi.OutputFlag'] = 0
        # ---- Create solver ----
        prob = {'f': self.opt.f, 'x': cs.vertcat(*self.opt.x), 'g': cs.vertcat(*self.opt.g), 'p': cs.vertcat(*self.opt.p)}
        if (solver_name == 'ipopt') or (solver_name == 'snopt'):
            self.solver = cs.nlpsol('solver', solver_name, prob, opts_dict)
            if code_gen:
                if not os.path.isfile('./' + prog_name + '.so'):
                    self.solver.generate_dependencies(prog_name + '.c')
                    os.system('gcc -fPIC -shared -O3 ' + prog_name + '.c -o ' + prog_name + '.so')
                self.solver = cs.nlpsol('solver', solver_name, prog_name + '.so', opts_dict)
        elif (solver_name == 'gurobi') or (solver_name == 'qpoases'):
            self.solver = cs.qpsol('solver', solver_name, prob, opts_dict)
        #  -------------------------------------------------------------------

    def solveProblem(self, idx, x0):
        # ---- setting parameters ---- 
        p_ = []  # set to empty before reinitialize
        p_ += x0
        p_ += self.X_nom_val[:, idx:(idx+self.TH)].elements()
        # ---- Solve the optimization ----
        start_time = time.time()
        sol = self.solver(x0=self.args.x0, lbx=self.args.lbx, ubx=self.args.ubx, lbg=self.args.lbg, ubg=self.args.ubg, p=p_)
        # ---- save computation time ---- 
        t_opt = time.time() - start_time
        # ---- decode solution ----
        resultFlag = self.solver.stats()['success']
        opt_sol = sol['x']
        f_opt = sol['f']
        # get x_opt, u_opt, other_opt
        x_opt = []
        for i in range(self.dyn.Nx):
            x_opt = cs.vertcat(x_opt, opt_sol[i::self.Nopt].T)
        u_opt = []
        for i in range(self.dyn.Nx, self.Nxu):
            u_opt = cs.vertcat(u_opt, opt_sol[i::self.Nopt].T)
        other_opt = []
        for i in range(self.Nxu, self.Nopt):
            other_opt = cs.vertcat(other_opt, opt_sol[i::self.Nopt].T)
        # other_opt = opt_sol[self.Nxu::self.Nopt].elements()
        # ---- warm start ---- 
        # opt_sol[0::self.Nopt] = [0.0]*(self.TH)
        # opt_sol[1::self.Nopt] = [0.0]*(self.TH)
        # opt_sol[2::self.Nopt] = [0.0]*(self.TH)
        # opt_sol[3::self.Nopt] = [0.0]*(self.TH)
        # opt_sol[4::self.Nopt] = [0.0]*(self.TH-1)
        # opt_sol[5::self.Nopt] = [0.0]*(self.TH-1)
        opt_sol[6::self.Nopt] = [0.0]*(self.TH-1)
        opt_sol[7::self.Nopt] = [0.0]*(self.TH-1)
        opt_sol[8::self.Nopt] = [0.0]*(self.TH-1)
        self.args.x0 = opt_sol.elements()

        return resultFlag, x_opt, u_opt, other_opt, f_opt, t_opt
