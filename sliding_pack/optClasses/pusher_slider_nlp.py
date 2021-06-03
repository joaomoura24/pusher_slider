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

    def __init__(self, dyn_class, dt=0.1):

        # init parameters
        self.dyn = dyn_class

        # initialize variables for opt and args
        self.opt = sliding_pack.opt.OptVars()
        self.args = sliding_pack.opt.OptArgs()

#    def buildProblem(self):
#        # ...

#    def solveProblem(self):
#        # 

#    def decodeSol(self):
#        #
