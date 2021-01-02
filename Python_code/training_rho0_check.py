from os import error
import numpy as np
import deepxde as dde
from deepxde.backend import tf
from numpy.lib import polynomial
import scipy
import math
from scipy.special import factorial
import os
import matplotlib.pyplot as plt
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
        # self.sizeall = (self.N + 1)**2 * self.alpha_max
        self.sizeRho = (self.N + 1)**2
        self.G = self.g * (self.nc + 1)**0.5
C =ContantNumbers()
DtypeTF = tf.float64
Gamma=tf.constant(C.Gamma/ 1e12,dtype=DtypeTF)
Ntf = tf.constant(C.N,dtype=DtypeTF)
sizeRho = tf.constant(C.sizeRho,dtype=DtypeTF)
n_bar = tf.constant(C.n_bar,dtype=DtypeTF)
lamb = tf.constant(C.lamb,dtype=DtypeTF)
sigma = tf.constant(C.sigma,dtype=DtypeTF)
G = tf.constant(C.G/ 1e12,dtype=DtypeTF)
g = tf.constant(C.g,dtype=DtypeTF)
w_vib = tf.constant(C.w_vib/ 1e12,dtype=DtypeTF)
w = tf.constant(C.w,dtype=DtypeTF)
v= tf.constant(C.v,dtype=DtypeTF)
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

def getXHalf_real(m,n,N):
    """
    get the rho number of y vecter for half of rho real part

    0<n<m<N+1
    n       1 -> N
    m       2 -> N+1
    result  N+1 -> (N+1)(N+2)/2 -1

    n==m
    n       1 -> N+1
    m       1 -> N+1
    result  m-1
    """
    assert(m >= n)
    assert(m <= N + 1 and m >=1)
    assert(n <= N + 1 and n >=1)
    if m==n:
        return m-1
    elif m>n:
        return round((m-2)*(m-1)/2 + n + N )
def getXHalf_imag(m,n,N):
    """
    get the rho number of y vecter for half of rho imag part 
    n<m
    n       1 -> N
    m       2 -> N+1
    result  (N+2)* (N+1) /2 -> (N+1)^2 -1

    n==m
    n       1 -> N+1
    m       1 -> N+1
    result  m-1
    """
    assert(m >= n)
    assert(m <= N + 1 and m >=1)
    assert(n <= N + 1 and n >=1)
    if m==n:
        return m-1
    elif m>n:
        return round( (m-2)*(m-1)/2 + n + (N+2)* (N+1) /2 -1)
def delta(a, b):
    if(a == b):
        return int(1)
    else:
        return int(0)

def SLE_q(x,y,C):
    """
    SLE function with q
    """
    Drho = []
    
    for allindex in range(0 ,C.sizeRho):
        DrhoQ = Gamma / 2 * y[:,allindex:allindex+1] * Ntf
        for qs in range(0 , C.N ):
            DrhoQ = DrhoQ + Gamma / 2 * ( 
            x[:,qs:qs+1]
            * dde.grad.jacobian(y, x, i=allindex ,j = qs) 
           +  (0.5 + n_bar) 
           * dde.grad.hessian(y, x, i=qs , j = qs , component=allindex)
            )
           
        DrhoT = dde.grad.jacobian(y, x, i=allindex ,j = C.N ) 
        Drho.append( DrhoQ  - DrhoT) 
        # Drho.append(DrhoT)
    # for n in range(1 ,C.N + 1):
    #         #  n = 1: N
    #     for m in range(n + 1 , C.N + 2 ):
    #         # m = n + 1: N + 1
    #         X_imag = getXHalf_imag(m,n, C.N)
    #         X_real = getXHalf_real(m,n, C.N)

    #         # real part 
    #         indexAA =  getXHalf_imag(C.N + 1, m , C.N)
    #         AA = - (
    #             y[:,X_imag:X_imag+1] * ( w - v -lamb * w_vib * x[:,n-1:n] )
    #             - G * y[:,indexAA:indexAA+1]
    #         )
    #         DD = AA *x[:,n-1:n] *0
    #         if m == C.N+1:
    #             for k in range(1,C.N+1):
    #                 if k >n:
    #                     indexDD = getXHalf_imag(k,n,C.N)
    #                     DD = DD + G * y[:,indexDD:indexDD+1]
    #                 elif k < n:
    #                     indexDD = getXHalf_imag(n,k,C.N)
    #                     DD = DD - G * y[:,indexDD:indexDD+1]
    #         else:
    #             indexDD = getXHalf_imag(C.N+1, n ,C.N)
    #             DD = y[:,X_imag:X_imag+1] * \
    #                 (w - v - lamb * w_vib * x[:,m-1:m] ) \
    #                     + G * y[:, indexDD:indexDD+1]
            
    #         # imag part 
    #         indexEE = getXHalf_real(C.N+1,m,C.N )
    #         EE = y[:,X_real:X_real+1] * (w - v - lamb * w_vib * x[:, n-1:n]) \
    #             + G * y[:,indexEE:indexEE+1 ]
    #         HH = EE * 0
    #         if m == C.N +1 :
    #             for k in range(1,C.N + 1):
    #                 if k > n:
    #                     indexEE = getXHalf_real(k,n,C.N)
    #                     HH = HH - G * y[:,indexEE :indexEE +1]
    #                 elif k < n:
    #                     indexEE = getXHalf_real(n,k,C.N)
    #                     HH = HH - G * y[:,indexEE:indexEE+1]
    #                 elif k == n:
    #                     HH = HH - G * y[:,n-1:n]
    #         else:
    #             indexEE = getXHalf_real(C.N + 1,n,C.N)
    #             HH = - y[:, X_real:X_real+1]* \
    #                 (w - v - lamb * w_vib * x[:, m-1:m]) \
    #                     - G * y[:,indexEE:indexEE+1]
    #         Drho[X_real] = Drho[X_real] + AA + DD
    #         Drho[X_imag] = Drho[X_imag] + EE + HH

    return Drho

def phi(q,alphaS):
    """
    phi for position q_s and eigen value alpha_s
    NOte : H_alpha is the physical form.
    ψ_(α_s ) (q_s )=e^(-q_s^2/2σ^2 )/√(2^(α_s ) α_s !〖(2π)〗^(1/2) σ) H_(α_s ) (q_s/(√2 σ))
    """
    
    H_alpha = scipy.special.eval_hermite(round(alphaS),q/(2**0.5 * C.sigma))
    Phivalue= (np.exp(- q **2 / 2 / C.sigma**2)
     / np.sqrt(
        2 ** alphaS * factorial(alphaS, exact=True) 
        * (2*np.pi) ** 0.5 *C.sigma
    ) * H_alpha)
    return Phivalue
def initialState(X,wei=-1):
    """
    initial state for the rho 2020/12/25 wodnering
    ρ(q,0)=1/[〖(2π)〗^(1/4) √σ]^N |├ UP⟩⟨UP┤|∙∏_(s=1)^N▒〖ψ_((α_s=0)) (q_s)〗
    
    ρ(q,0)=1/[〖(2π)〗^(1/4) √σ]^N  ∑_(i=1)^(N+1)▒∑_(j=1)^(N+1)▒〖ρ_ij (0)〗|├ i⟩⟨j┤|∙∏_(s=1)^N▒〖ψ_((α_s=0)) (q_s)〗

    ρ_ij (0)=1/2N (1-δ_(i,N+1) )(1-δ_(j,N+1) )+1/2 δ_(i,N+1) δ_(j,N+1)
                        +1/(2√N) [(1-δ_(i,N+1) ) δ_(j,N+1)+δ_(i,N+1) (1-δ_(j,N+1) )]
    ψ_((α_s=0) ) (q_s )=e^(-q_s^2/2σ^2 )/(〖(2π)〗^(1/4) √σ) H_0 (q_s/(√2 σ))
    """
    def rhoij(i,j):
        """
        get the rho(i,j) for inital state
        """
        assert(i >= j)
        if i == j:
            if i == N+1:
                Rhoij = 1/2
            else:
                Rhoij = 1/2/C.N
        elif i == N+1:
            Rhoij = 1 / 2/ math.sqrt(C.N)
        else:
            Rhoij = 1/2/C.N 
        return Rhoij
    NUM_initial,_ = np.shape(X)
    Rho0 = np.zeros([NUM_initial,C.sizeRho])
    N = C.N
    ConstantNumber = 1/ ((2*np.pi)**(1/4) * np.sqrt(C.sigma) )**N
    Phi0_multi = 1
    for s in range(0,N):
        Phi0_multi = Phi0_multi * phi(X[:,s:s+1] , 0) 
    for j in range(1,C.N + 2):
        for i in range(j + 1 ,C.N +2):
            Xnum = getXHalf_real(i,j,C.N)
            X_imag = getXHalf_imag(i,j,C.N) 
            Rho0[:,Xnum:Xnum+1] =ConstantNumber * Phi0_multi * rhoij(i,j) 

            Rho0[:,X_imag:X_imag+1] = 0
    for n  in range(1,N + 2):
        Rho0[:,n-1:n] =ConstantNumber * Phi0_multi * rhoij(n,n)
    if wei == -1:
        return Rho0
    else:
        return Rho0[:,wei:wei+1]
def boundary(_, on_initial):
    return on_initial
def plot_initial():
    x0 = np.random.random([500,C.N+1])
    Qmax = 10/C.sigma
    x0[:,0:C.N] = 2 * Qmax*(x0[:,0:C.N]-0.5)
    x0[:,1]=x0[:,1]*0
    y0 = initialState(x0)
    for i in range(0,9):
        plt.subplot(3,3,i+1)
        plt.plot(x0[:,0],y0[:,i],'.')
    plt.show()

def main():
    """
    main function for differential function with q
    """
    
    dde.config.real.set_float64()
    # geometry part


    tmax = 1
    Qmax = 10/C.sigma
    Xmin=[]
    Xmax = []
    for i in range(1,C.N + 1):
        Xmin.append(-Qmax)
        Xmax.append(Qmax)
    x0 = np.random.random([1000,C.N+1])
    xtest = np.random.random([1000,C.N+1])
    ytest = initialState(xtest)
    x0[:,0:C.N] = 2 * Qmax*(x0[:,0:C.N]-0.5)
    x0[:,1]=x0[:,1]*0
    x0[:,2]=x0[:,2]*0
    geom = dde.geometry.Hypercube(Xmin,Xmax)
    # geom = dde.geometry.Interval(-1, 1)
    ob_y = initialState(x0)
    '''
    check rho0
        for i in range(0,len(ob_y[1,:])):
            for j in range(0,len(x0[1,:])):
                plt.plot(x0[:,j],ob_y[:,i],'.')
                plt.show()
    '''
    timedomain = dde.geometry.TimeDomain(0, tmax)
    geom = dde.geometry.GeometryXTime(geom, timedomain)
    
    # Initial conditions
    ic = []
    ic_nb=[]
    ptset = dde.bc.PointSet(x0)
    
    inside = lambda x, _: ptset.inside(x)

    for j in range(0,(C.N+1) **2):
        ic.append(dde.IC(geom, lambda X: initialState(X,j), boundary,component= j))
        ic.append( dde.DirichletBC(
        geom, ptset.values_to_func(ob_y[:, j:j+1]), inside, component=j
        ))
        ic_nb.append(dde.IC(geom, lambda X: initialState(X,j), boundary,component= j))
        # test
        # print(initialState(x_initial,j))
    # print(SLE_q(xtest,ytest))
    # bc = dde.DirichletBC(geom,lambda _: 0, lambda _, on_boundary: on_boundary)
    # ic.append(bc)

    # data
    data1 = dde.data.TimePDE(geom,
    lambda x,y: SLE_q(x,y,C),
    ic_nb ,
    num_domain= 1000, 
    num_boundary= 0,
    num_initial=100, 
    num_test=None)
    data = dde.data.TimePDE(geom,
    lambda x,y: y*0,
    ic ,
    num_domain= 0, 
    num_boundary= 0,
    num_initial=100, 
    anchors=x0,
    num_test=None)
    # lambda x,y: SLE_q(x,y,C),
    # settings for model
    
    layer_size = [C.N+1] + [50] * 3 + [(C.N+1) **2]
    activation = "tanh"
    initializer = "Glorot uniform"


    save_model_dir = os.path.join(os.getcwd(),'Model_save','test_rho0')
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)
    save_model_name = os.path.join(os.getcwd(),'Model_save','test_rho0','test1230')
    load_epoch = '60000'
    load_model_name = os.path.join(os.getcwd(),'Model_save','test_rho0','test1231-'+load_epoch)
    Callfcn = dde.callbacks.ModelCheckpoint(save_model_name,verbose=1,save_better_only=True,period=10000)

    # initialize model
    net = dde.maps.FNN(layer_size, activation, initializer)
    model = dde.Model(data1,net)
    # load model
    model.compile("adam" , lr= 0.001)
    model.restore(load_model_name)

    y_pre = model.predict(x0)
    for i in range(0,9):
        plt.subplot(3,3,i+1)
        plt.plot(x0[:,0],ob_y[:,i],'.b')
        plt.plot(x0[:,0],y_pre[:,i],'.r')
    plt.show()

if __name__ == "__main__":
    plot_initial()
    # main()
