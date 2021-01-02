import os
import numpy as np
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
        self.e = 2.718281828459045
        self.Qmax = 10/self.sigma

def get_fileName_newest(test_report):
    lists = os.listdir(test_report) #列出目录的下所有文件和文件夹保存到lists
    print(list)
    lists.sort(key=lambda fn:os.path.getmtime(test_report + "\\" + fn))  # 按时间排序
    file_new = os.path.join(test_report,lists[-1]) # 获取最新的文件保存到file_new
    print(file_new)

    return file_new.split('.')[0]
def delta(a, b):
    if(a == b):
        return int(1)
    else:
        return int(0)

def load_model_name(floderName,epoch = -1,modelName=None):
    """
    load a trained model in the specific filefolder
    """
    if epoch == -1:
        Folder = os.path.join(os.getcwd(),'Model_save',floderName)

        Name = get_fileName_newest(Folder)
    else:
        Name = os.path.join(os.getcwd(),'Model_save',floderName,modelName)+ str(round(epoch))
    return Name
def get_ij(n,N):
    """
        get i and j from n
    """
    if n < N+1:
        i = n+1
        j = n+1
def boundary(_, on_initial):
    return on_initial
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
