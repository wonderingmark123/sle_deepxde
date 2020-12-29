from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import deepxde as dde
from deepxde.backend import tf
from SLE_functions import *

global C
# initialize some parameters for the following caculation.
C = ContantNumbers()


def SLE_DL(t, y):
    """
    For the deep learning method
    """
    # DyFun = SLEfun(y,C)
    DyFun = tf.numpy_function(SLEfun,[y],tf.complex128)
    Dygrand = DyFun*0
    for i in range(0,C.sizeall):
        Dygrand[i] = dde.grad.jacobian(y,t,i=i)
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
    maxX = getX(C.N + 1,C.N+1,C.alpha_max,C)
    y0 = np.zeros(maxX)
    for i in range(0, C.N+2):
        for j in range(0, C.N+2):
            for alpha in [1]:

                X = getX(i, j, alpha,C)

                y0[X] = 1./2./C.N * (1-delta(i, C.N+1))*(1-delta(j, C.N+1))+1./2*delta(i, C.N+1)*delta(j, C.N+1) +\
                    1./2./(C.N)**0.5 * ((1-delta(i, C.N+1)) *
                                        delta(j, C.N+1)+(1-delta(j, C.N+1))*delta(i, C.N+1))
    return y0

def boundaryFunction(_, out_net, X):
    """
    boundary Function for SLE without q
    """
    return out
def main():
    """
    main program
    haotian song 2020/12/23 
    """
    geom = dde.geometry.TimeDomain(0, 3)
    y0 = InitialCondition()
    a = dde.config.real.set_complex128()
    ic = []
    for j in range(0,len(y0)):
        ic.append(dde.IC(geom, lambda X: y0, boundary,component= j))
    # BoundaryCondition = dde.OperatorBC(geom,boundaryFunction,on_boundary=on_boundary)
    data = dde.data.PDE(geom, SLE_DL, ic ,num_domain= 400, num_boundary= 2, num_test=100)
    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)
    model = dde.Model(data,net)
    model.compile("adam" , lr= 0.001)
    variable = dde.callbacks.VariableValue(
        [C1, C2, C3], period=600, filename="variables.dat"
    )
    losshistory, train_state = model.train(epochs=60000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    # dde.PeriodicBC()
    # for i in range(0,C)
    # ic3 = dde.IC(geom, lambda X: 27, boundary, component=5)
    # data = dde.data.PDE(geom,)

if __name__ == "__main__":
    # TEST()
    main()
