from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import random
import deepxde as dde
from deepxde.backend import tf


class ContantNumbers:
    """
    some constant numbers for the whole program
    """

    def __init__(self):
        self.N = 2
        #  % number of atoms

        self.nc = 0
        self.cutoff = 9
        # % alpha_s cutoff, alpha_s=0,1,2,\,cutoff
        alpha_max = (self.cutoff+1)**self.N
        # % defined to instead alpha_1, alpha_2, \, alpha_N
        self.alpha_max = alpha_max
        # % Parameter set:
        c = 3*10**8
        self.w = 0
        self.v = 0
        w_vib = 1200
        self.w_vib = w_vib*100*c
        # % [cm^(-1)] ---> [m^(-1)]  ---> s^ -1
        self.Gamma = 33 * 100*c
        # % [cm^(-1)]
        self.lamb = 0.5
        self.g = 70*100*c
        # % [cm^(-1)]
        # self.alphaArray = reshape(1:self.alpha_max,self.cutoff + 1,self.cutoff +1)
        n_c = 0
        # % ground state photon number |n_c>
        T = 300
        # % [K]
        hbar = 1.0545718*10**(-34)
        # % [m^2 * kg / s]
        k_b = 1.38064852*10**(-23)
        # % [m^2 * kg * s^(-2) * K^(-1)]

        self.n_bar = (np.exp(hbar*self.w_vib/k_b/T)-1)**(-1)
        self.sigma = (self.n_bar+0.5)**0.5


# initialize some parameters for the following caculation.
C = ContantNumbers()


def getX(m, n, alphaNUM):
    """
    get the index of vecter using m n alphaNUm 
    """
    m = np.array(m)
    n = np.array(n)
    Xnum = m + (C.N+1)**1*(n-1) + (C.N+1)**2 * (alphaNUM-1)
    return Xnum-1


def getAlpha(alphaNUM):
    """
    get the index of individual alpha with alphaNUM
    """
    alphaNUM = alphaNUM - 1
    alpha = np.zeros([C.N], dtype=int)
    for i in range(1, C.N + 1):
        alpha[i - 1] = alphaNUM % (C.cutoff+1)
        alphaNUM = np.round((alphaNUM - alpha[i-1]) / (C.cutoff+1))
    return alpha


def SLEfun(y):
    
    Drho = np.zeros(np.size(y), dtype=complex)
    for m in range(1, C.N+2):
        for n in range(1, C.N+2):
            for alpha in range(1, C.alpha_max + 1):
                Xnum = getX(m, n, alpha)
                alphaList = getAlpha(alpha)
                Drho[Xnum] = - y[Xnum] * \
                    np.sum(alphaList) * C.Gamma / 2

                if m == C.N + 1:

                    #  sumINDEX = 1:C.N +  (C.N+1)**1*(n-1) + (C.N+1)**2 * (alpha-1)
                    sumINDEX = getX(range(1, C.N+1), n, alpha)
                    Drho[Xnum] = Drho[Xnum] + \
                        - 1j*C.g * np.sqrt(C.nc + 1) * np.sum(y[sumINDEX])
                else:
                    Drho[Xnum] = Drho[Xnum] + \
                        - 1j * C.g*np.sqrt(C.nc+1) * y[getX(C.N+1, n, alpha)]
                    alpham = alphaList[m - 1]
                    INDplus = getX(m, n, alpha + (C.cutoff + 1)**(m-1))
                    INDminus = getX(m, n, alpha - (C.cutoff + 1)**(m-1))
                    if(alpham < C.cutoff):
                        Drho[Xnum] = Drho[Xnum] + \
                            1j * C.lamb * C.w_vib * C.sigma * \
                            np.sqrt(alpham + 1) * y[INDplus]

                    if(alpham > 0):
                        Drho[Xnum] = Drho[Xnum] + \
                            1j * C.lamb * C.w_vib * C.sigma * \
                            np.sqrt(alpham) * y[INDminus]

                if n == C.N + 1:
                    sumINDEX = getX(m, range(1, C.N+1), alpha)
                    Drho[Xnum] = Drho[Xnum] + \
                        1j*C.g*np.sqrt(C.nc + 1) * np.sum(y[sumINDEX])
                else:
                    Drho[Xnum] = Drho[Xnum] + \
                        1j * C.g*np.sqrt(C.nc + 1) * y[getX(m, C.N+1, alpha)]
                    alpham = alphaList[n - 1]
                    INDplus = getX(m, n, alpha + (C.cutoff + 1)**(n-1))
                    INDminus = getX(m, n, alpha - (C.cutoff + 1)**(n-1))
                    if(alpham < C.cutoff):
                        Drho[Xnum] = Drho[Xnum] + \
                            -1j * C.lamb * C.w_vib * C.sigma * \
                            np.sqrt(alpham + 1) * y[INDplus]

                    if(alpham > 0):
                        Drho[Xnum] = Drho[Xnum] + \
                            -1j * C.lamb * C.w_vib * C.sigma * \
                            np.sqrt(alpham) * y[INDminus]

    # % Drho = Drho/1d12
    return Drho

def SLE_DL(t, y):
    """
    For the deep learning method
    """
    DyFun = SLEfun(y)
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
    dy = SLEfun(yMAT)
    a = dy - DyMat[0]
    print(np.shape(dy))
    print(np.shape(DyMat))
    print(np.shape(a))
    print(a)
    return a

def boundary(_, on_initial):
    return on_initial


def delta(a, b):
    if(a == b):
        return 1
    else:
        return 0


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
    # ic3 = dde.IC(geom, lambda X: 27, boundary, component=5)
    data = dde.data.PDE(
        geom,
        Lorenz_system,
        [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
        num_domain=400,
        num_boundary=2,
        anchors=observe_t,
    )

if __name__ == "__main__":
    TEST()
    main()
