## Author: Joao Moura
## Contact: jpousad@ed.ac.uk
## Date: 19/10/2020
## -------------------------------------------------------------------
## Description:
## 
## classes for facilitating the optimization definition
## -------------------------------------------------------------------

## Define structures for optimization variables and optimization arguments
#  -------------------------------------------------------------------
class OptVars():
    x = None # optimization independent variables
    g = None # optimization equality constraints
    p = None # optimization parameters
    f = None # optimization cost
    discrete = None # flag for indicating integer variables
class OptArgs():
    x0 = None # initial guess for optimization independent varibles
    p = None # parameters
    lbg = None # lower bound for for constraint g
    ubg = None # upper bound for the constraint g
    lbx = None # lower bound for optimization variables
    ubx = None # upper bound for optimization variables
#  -------------------------------------------------------------------
