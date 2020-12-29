from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from deepxde import initial_condition
import numpy as np
import deepxde as dde
from deepxde.backend import tf
from SLE_functions import *
import scipy as sp
import scipy.interpolate as interp
from scipy.integrate import odeint
global C
# initialize some parameters for the following caculation.
C = ContantNumbers()


def SLE_DL(t, y):
    """
    For the deep learning method
    """
    DyFun = SLEfun(y,C)
    Dygrand = tf.gradients(y, t)[0]
    return Dygrand - DyFun


def TEST():
    """
    test SLEfun with matalb results
    """
    # test
    yMAT = np.load('./yrand900MATLAB.npy')
    DyMat = np.load('./Dyrand900MATLABreal.npy') + 1j * \
        np.load('./Dyrand900MATLABimag.npy')
    # y = rand(900,1)
    dy = SLEfun(yMAT,C)
    a = dy - DyMat[0]
    print(np.shape(dy))
    print(np.shape(DyMat))
    print(np.shape(a))
    print(a)
    return a

def boundary(_, on_initial):
    return on_initial



def InitialCondition():
    """
    initial function of rho 
    return a vector.
    """
    maxX = getX(C.N + 1,C.N+1,C.alpha_max)
    y0 = np.zeros(maxX,dtype=complex)
    for i in range(0, C.N+2):
        for j in range(0, C.N+2):
            for alpha in [1]:

                X = getX(i, j, alpha)

                y0[X] = 1./2./C.N * (1-delta(i, C.N+1))*(1-delta(j, C.N+1))+1./2*delta(i, C.N+1)*delta(j, C.N+1) +\
                    1./2./(C.N)**0.5 * ((1-delta(i, C.N+1)) *
                                        delta(j, C.N+1)+(1-delta(j, C.N+1))*delta(i, C.N+1))


def main():
    """
    main program
    haotian song 2020/12/23 
    """
    geom = dde.geometry.TimeDomain(0, 3)
    y0 = initial_condition()
    for i in range(0,C)
    # ic3 = dde.IC(geom, lambda X: 27, boundary, component=5)
    data = dde.data.PDE(
        geom,
        SLE_DL,
        [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
        num_domain=400,
        num_boundary=2,
        anchors=observe_t,
    )
    

if __name__ == "__main__":
    TEST()
    # main()
