from os import error
import numpy as np
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
        self.sizeall = (self.N + 1)**2 * self.alpha_max


def getX(m, n, alphaNUM,C):
    """
    get the index of vecter using m n alphaNUm 
    """
    m = np.array(m)
    n = np.array(n)
    Xnum = m + (C.N+1)**1*(n-1) + (C.N+1)**2 * (alphaNUM-1)
    return Xnum-1


def getAlpha(alphaNUM,C):
    """
    get the index of individual alpha with alphaNUM
    """
    alphaNUM = alphaNUM - 1
    alpha = np.zeros([C.N], dtype=int)
    for i in range(1, C.N + 1):
        alpha[i - 1] = alphaNUM % (C.cutoff+1)
        alphaNUM = np.round((alphaNUM - alpha[i-1]) / (C.cutoff+1))
    return alpha


def SLEfun(y,C):
    Drho = y*0
    for m in range(1, C.N+2):
        for n in range(1, C.N+2):
            for alpha in range(1, C.alpha_max + 1):
                Xnum = getX(m, n, alpha,C)
                alphaList = getAlpha(alpha,C)
                Drho[Xnum] = - y[Xnum] * \
                    np.sum(alphaList) * C.Gamma / 2

                if m == C.N + 1:

                    #  sumINDEX = 1:C.N +  (C.N+1)**1*(n-1) + (C.N+1)**2 * (alpha-1)
                    sumINDEX = getX(range(1, C.N+1), n, alpha,C)
                    Drho[Xnum] = Drho[Xnum] + \
                        - 1j*C.g * np.sqrt(C.nc + 1) * np.sum(y[sumINDEX])
                else:
                    Drho[Xnum] = Drho[Xnum] + \
                        - 1j * C.g*np.sqrt(C.nc+1) * y[getX(C.N+1, n, alpha,C)]
                    alpham = alphaList[m - 1]
                    INDplus = getX(m, n, alpha + (C.cutoff + 1)**(m-1) ,C)
                    INDminus = getX(m, n, alpha - (C.cutoff + 1)**(m-1),C)
                    if(alpham < C.cutoff):
                        Drho[Xnum] = Drho[Xnum] + \
                            1j * C.lamb * C.w_vib * C.sigma * \
                            np.sqrt(alpham + 1) * y[INDplus]

                    if(alpham > 0):
                        Drho[Xnum] = Drho[Xnum] + \
                            1j * C.lamb * C.w_vib * C.sigma * \
                            np.sqrt(alpham) * y[INDminus]

                if n == C.N + 1:
                    sumINDEX = getX(m, range(1, C.N+1), alpha,C)
                    Drho[Xnum] = Drho[Xnum] + \
                        1j*C.g*np.sqrt(C.nc + 1) * np.sum(y[sumINDEX])
                else:
                    Drho[Xnum] = Drho[Xnum] + \
                        1j * C.g*np.sqrt(C.nc + 1) * y[getX(m, C.N+1, alpha,C)]
                    alpham = alphaList[n - 1]
                    INDplus = getX(m, n, alpha + (C.cutoff + 1)**(n-1),C)
                    INDminus = getX(m, n, alpha - (C.cutoff + 1)**(n-1),C)
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

def delta(a, b):
    if(a == b):
        return 1
    else:
        return 0
def getXHalf(m,n,N):
    """
    get the rho number of y vecter for half of rho
    """
    assert(n<m)

    return m*(m-1)/2 + n -1
def SLE_q(y,C):
    """
    SLE function with q
    """
    for m in range(0 ,C.N + 1):
        for n in range(m, C.N + 1 ):
