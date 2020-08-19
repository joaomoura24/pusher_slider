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
int_square = lambda a: scipy.integrate.dblquad(lambda x,y: numpy.square(x**2 + y**2), -a/2, a/2, -a/2, a/2)[0]
int_A = int_square(a)
m_max = miu_g*m*g*int_A/A
#  -------------------------------------------------------------------

## Build Motion Model
#  -------------------------------------------------------------------
#  -------------------------------------------------------------------
