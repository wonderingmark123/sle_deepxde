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
from utils import *
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
pi = tf.constant(np.pi,dtype=DtypeTF)
etf = tf.constant(C.e,dtype=DtypeTF)
ConstantNumber = 1/ ((2*pi)**(1/4) * (sigma)**0.5 )**Ntf
QmaxTF = tf.constant(C.Qmax,dtype=DtypeTF)
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

    return tf.concat(Drho, axis=1)

def phi(q,alphaS):
    """
        phi for position q_s and eigen value alpha_s
        NOte : H_alpha is the physical form.
        ψ_(α_s ) (q_s )=e^(-q_s^2/2σ^2 )/√(2^(α_s ) α_s !〖(2π)〗^(1/2) σ) H_(α_s ) (q_s/(√2 σ))
    """
    if(alphaS == 0):
        Phivalue= etf ** (- q **2 / (2 * sigma**2)) / (
          (2* pi) ** 0.5 *sigma
                )**0.5
    else:
        H_alpha = scipy.special.eval_hermite(round(alphaS),q/(2**0.5 * C.sigma))
        Phivalue= etf ** (- q **2 / 2 / sigma**2) / (
        2 ** alphaS * factorial(alphaS, exact=True) * (2*np.pi) ** 0.5 *C.sigma
                )**0.5 * H_alpha
    return Phivalue
def initialState_tf(X,wei=-1,sample_output=None):
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
                Rhoij = 1/2/Ntf
        elif i == N+1:
            Rhoij = 1 / 2/ (Ntf)**0.5
        else:
            Rhoij = 1/2/Ntf
        return Rhoij
    
    if sample_output is None:
        NUM_initial,_ = np.shape(X)
        Rho0 = np.zeros([NUM_initial,C.sizeRho])
    else:
        Rho0 = []
        for i in range(0,C.sizeRho):
            Rho0.append(sample_output[:,i:i+1]*0)
    N = C.N
    ConstantNumber = 1/ ((2*pi)**(1/4) * (sigma)**0.5 )**Ntf
    Phi0_multi = 1
    for s in range(1,N):
        Phi0_multi = Phi0_multi * phi(X[:,s:s+1] , 0) 
    for j in range(1,C.N + 2):
        for i in range(j + 1 ,C.N +2):
            Xnum = getXHalf_real(i,j,C.N)
            X_imag = getXHalf_imag(i,j,C.N) 
            Rho0[Xnum] =Rho0[Xnum] + ConstantNumber * Phi0_multi * rhoij(i,j)

    for n  in range(1,N + 2):
        Rho0[n-1] =Rho0[n-1]+ ConstantNumber * Phi0_multi * rhoij(n,n)
    if wei == -1:
        return tf.concat(Rho0, axis=1)
    else:
        return Rho0[:,wei:wei+1]
def boundary(_, on_initial):
    return on_initial
def boundary_condition(x,y):
    """
        main function for differential function with q
    """
    def rhoij(i,j):
        """
        get the rho(i,j) for inital state
        """
        assert(i >= j)
        if i == j:
            if i == C.N+1:
                Rhoij = 1/2
            else:
                Rhoij = 1/2/Ntf
        elif i == C.N+1:
            Rhoij = 1 / 2/ (Ntf)**0.5
        else:
            Rhoij = 1/2/Ntf
        return Rhoij
    Rho0 = []
    for i in range(0,C.sizeRho):
        Rho0.append(y[:,i:i+1]*0)
        
    Phi0_multi = etf ** (- x[:,0:1] **2 / (2 * sigma**2)) / (
          (2* pi) ** 0.5 *sigma
                )**0.5
    for s in range(1,C.N):
        Phi0_multi = Phi0_multi * etf ** (- x[:,s:s+1] **2 / (2 * sigma**2)) / (
          (2* pi) ** 0.5 *sigma
                )**0.5
    for n  in range(1,C.N + 2):
        Rho0[n-1] = Rho0[n-1] + ConstantNumber * Phi0_multi * rhoij(n,n)
    for j in range(1,C.N+1):
        for i in range(j+1,C.N+1):
            Xnum = getXHalf_real(i,j,C.N)
            Rho0[Xnum] = Rho0[Xnum] * ConstantNumber * Phi0_multi * rhoij(i,j)
    result = []
    for i in range(0,C.sizeRho):
        result.append(x[:, C.N:] * y[:,i:i+1] + Rho0[i])
        for j in range(0,C.N):
            result[i] = result[i] * (x[:,j:j+1] - QmaxTF)*(
                x[:,j:j+1] + QmaxTF)
    return tf.concat(result, axis=1)
def main():
    """
        main function for differential function with q
    """
    
    dde.config.real.set_float64()
    # geometry part


    tmax = 1
    
    Qmax = C.Qmax
    Xmin=[]
    Xmax = []
    for i in range(1,C.N + 1):
        Xmin.append(-Qmax)
        Xmax.append(Qmax)

    geom = dde.geometry.Hypercube(Xmin,Xmax)
    # geom = dde.geometry.Interval(-1, 1)
    # ob_y = initialState(x0)
    '''
    check rho0
        for i in range(0,len(ob_y[1,:])):
            for j in range(0,len(x0[1,:])):
                plt.plot(x0[:,j],ob_y[:,i],'.')
                plt.show()
    '''
    timedomain = dde.geometry.TimeDomain(0, tmax)
    geom = dde.geometry.GeometryXTime(geom, timedomain)
    


        # test
        # print(initialState(x_initial,j))
    # print(SLE_q(xtest,ytest))
    # bc = dde.DirichletBC(geom,lambda _: 0, lambda _, on_boundary: on_boundary)
    # ic.append(bc)

    # data
    data_sle = dde.data.TimePDE(geom,
    lambda x,y: SLE_q(x,y,C),
    [] ,num_domain= 1000)

    # lambda x,y: SLE_q(x,y,C),
    # settings for model
    
    layer_size = [C.N+1] + [50] * 3 + [(C.N+1) **2]
    activation = "tanh"
    initializer = "Glorot uniform"


    save_model_dir = os.path.join(os.getcwd(),'Model_save','test_rho0_0101')
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)
    save_model_name = os.path.join(os.getcwd(),'Model_save','test_rho0','test1230')
    load_epoch = '60000'
    load_model_name = os.path.join(os.getcwd(),'Model_save','test_rho0','test1230-'+load_epoch)
    Callfcn = dde.callbacks.ModelCheckpoint(save_model_name,verbose=1,save_better_only=True,period=10000)
    
    # initialize model
    net = dde.maps.FNN(layer_size, activation, initializer)
    # net.apply_output_transform(boundary_condition)
    model = dde.Model(data_sle,net)
    
    model.compile("adam" , lr= 0.001, metrics=["l2 relative error"])
    # model.restore(load_model_name)
    # model_sle = dde.Model(data_sle,model.net)
    # model_sle.compile("adam" , lr= 0.001)
    losshistory, train_state = model.train(
        epochs=60000,
        callbacks=[Callfcn],
        model_save_path=save_model_name
        )
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

if __name__ == "__main__":
    main()
