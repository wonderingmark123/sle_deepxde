from os import error
import numpy as np
import deepxde as dde
from deepxde.backend import tf
from numpy.core.function_base import linspace
from numpy.lib import polynomial
import scipy
import math
from scipy.special import factorial
import os
import matplotlib.pyplot as plt
from utils import *

C =ContantNumbers()
DtypeTF = tf.float32
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
    Phivalue= np.exp(- q **2 / 2 / C.sigma**2) / np.sqrt(
        2 ** alphaS * factorial(alphaS, exact=True) * (2*np.pi) ** 0.5 *C.sigma
    ) * H_alpha
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
def plot_3d_initial():
    fig = plt.figure()  #定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')
    Qmax = 10/C.sigma
    #定义三维数据
    num=1000
    x0 = np.random.random([num,C.N+1])
    x0[:,0:C.N] = 2 * Qmax*(x0[:,0:C.N]-0.5)
    zz = initialState(x0)
    for i in range(0,len(zz[1,:])):
        X, Y,Z = np.meshgrid(x0[:,0], x0[:,1],zz[:,1])
        # Z = np.sin(X)+np.cos(Y)


        #作图
        ax3.plot_surface(X,Y,Z,cmap='rainbow')
        #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
        plt.show()  
def initial_condition(x,y):
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
def boundary_condition(x,y):
    """
        exact boundary conditions
    """
    # re = []
    # for i in range(0,C.sizeRho):
    #     re.append(y[:,i:i+1] * (x[:,0:1]**2 - QmaxTF**2) 
    #     * (x[:,1:2]**2 + QmaxTF**2))
    # return tf.concat(re,axis=1)
    return y * (x[:,0:1]**2 - QmaxTF**2)
    
def main():
    """
    main function for differential function with q
    """
    
    # dde.config.real.set_float64()
    # geometry part
    tmax = 1
    Qmax = 10/C.sigma
    Xmin=[]
    Xmax = []

    for i in range(1,C.N + 1):
        Xmin.append(-Qmax)
        Xmax.append(Qmax)
    
    geom = dde.geometry.Hypercube(Xmin,Xmax)
    # geom = dde.geometry.Interval(-1, 1)
    
    '''
    check rho0
        for i in range(0,len(ob_y[1,:])):
            for j in range(0,len(x0[1,:])):
                plt.plot(x0[:,j],ob_y[:,i],'.')
                plt.show()
    '''
    
    timedomain = dde.geometry.TimeDomain(0, tmax)
    geom = dde.geometry.GeometryXTime(geom, timedomain)

    # test points setting
    # x0 = np.random.random([1000,C.N+1])
    # x0[:,0:C.N] = 2 * Qmax*(x0[:,0:C.N]-0.5)
    # ob_y = initialState(x0)
    # ptset = dde.bc.PointSet(x0)
    # inside = lambda x, _: ptset.inside(x)

    # x_initial = np.random.rand(13, C.N+1)
    # xtest = tf.convert_to_tensor(x_initial)
    # ytest = np.random.rand(13, C.sizeRho)
    # ytest = tf.convert_to_tensor(ytest)

    # Initial conditions
    ic = []


    for j in range(0,(C.N+1) **2):
        ic.append(dde.IC(geom, lambda X: initialState(X,j), boundary,component= j))
        # test
        # print(initialState(x_initial,j))
    # print(SLE_q(xtest,ytest))
    # bc = dde.DirichletBC(geom,lambda _: 0, lambda _, on_boundary: on_boundary)
    # ic.append(bc)

    # data
    data = dde.data.TimePDE(geom,
    lambda x,y: SLE_q(x,y,C),
    ic ,
    num_domain= 2000, 
    num_boundary= 0,
    num_initial=500, 
    num_test=None)

    # settings for model
    
    layer_size = [C.N+1] + [100] * 5 + [(C.N+1) **2]
    activation = "tanh"
    initializer = "Glorot uniform"


    save_model_dir = os.path.join(os.getcwd(),'Model_save','test_rho0_0102')
    if not os.path.isdir(save_model_dir):
        os.mkdir(save_model_dir)
    save_model_name = os.path.join(save_model_dir,'model_save')
    load_epoch = '15000'
    load_model_name = save_model_name + '-' + load_epoch
    Callfcn = dde.callbacks.ModelCheckpoint(save_model_name,verbose=1,save_better_only=True,period=10000)

    # initialize model
    net = dde.maps.FNN(layer_size, activation, initializer)
    net.apply_output_transform(boundary_condition)
    model = dde.Model(data,net)
    
    model.compile("adam" , lr= 0.001)
    model.restore(load_model_name)
    
    losshistory, train_state = model.train(epochs=60000,callbacks=[Callfcn],model_save_path=save_model_name)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

if __name__ == "__main__":
    # plot_3d_initial()
    main()
