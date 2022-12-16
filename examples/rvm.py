import numpy as np
from numpy.linalg import inv, det
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from stateml.gpr import Util

def basis_function(x) :
    types = 'gaussian'#sys.argv[1]
    global M
    phi = np.ones((M))
    if (types == 'polynomial'):
        for m in range(0, M):
            phi[m] = x**m

    elif (types == 'gaussian'):
        myu = np.linspace(-3,4,M)
        s = 1.0
        for m in range(0, M):
            phi[m] = np.exp(-(x-myu[m])**2/(2*s**2))

    elif (types == 'sigmoidal'):
        myu = np.linspace(-5,5,M)
        s = 0.1
        for m in range(0, M):
            phi[m] = 1/(1+np.exp(-(x-myu[m])/s))

    return (phi)

def design_matrix(data):
    global M
    N = len(data)
    PHI = np.ones((N,M))
    for n in range(0,N):
        PHI[n] = basis_function(data[n])

    return (PHI)

def get_mean(beta, Sigma, PHI, t):
    mean = beta * Sigma @ PHI.T @ t

    return mean

def get_Sigma(beta, A, PHI):
    Sigma = inv(A + beta*PHI.T @ PHI)

    return Sigma

def get_Phi(design_matrix):
    global M
    Phi = []
    for i in range(M):
        Phi.append(design_matrix.T[i])
    Phi = np.array(Phi)
    
    return Phi

def get_Covariance(beta, PHI, A):
    global N
    I = np.identity(N)
    C = 1/beta * I + PHI @ inv(A) @ PHI.T

    return C

def get_s_vec(Phi, C_mod):
    global M
    s_vec = np.zeros(M)
    for i in range(M):
        s_vec[i] = Phi[i].T @ inv(C_mod) @ Phi[i]

    return s_vec

def get_q_vec(Phi, C_mod, t):
    global M
    q_vec = np.zeros(M)
    for i in range(M):
        q_vec[i] = Phi[i].T @ inv(C_mod) @ t

    return q_vec

def func(X):
    #y = np.sin(X) + X
    y = np.sin(3*X + 1) + X
    #y = np.sin(2*X**2+1) + X

    return y

#####
X_true = np.arange(-3, 4,0.01)
t_true = func(X_true)
noise = 0.25
X_train = np.arange(-3, 4, 1)
t_train = func(X_train) + noise * np.random.randn(*X_train.shape)

X_test = np.arange(-3, 4, 0.2)
#####

#####
M = 8
N = len(t_train)
PHI = design_matrix(X_train)
beta_init = 0.1
inf = 1e9
alphas_init = np.zeros(M) + inf
alphas_init[0] = 1
max_iter = 1000
#####

Phi = get_Phi(PHI)

for iternum in range(max_iter):
    print (iternum)
    beta = beta_init
    alphas = alphas_init
    A = np.diag(alphas)
    Sigma = get_Sigma(beta, A, PHI)
    mean  = get_mean(beta, Sigma, PHI, t_train)
    C     = get_Covariance(beta, PHI, A)
    for i in range(M):
#        print ('basis_func:', i)
        C_mod = C - 1/alphas[i] * Phi[i] @ Phi[i].T
        q_vec = get_q_vec(Phi, C_mod, t_train)
        s_vec = get_s_vec(Phi, C_mod)
        if (q_vec[i]**2 > s_vec[i] and alphas[i] < inf):
            alphas[i] = s_vec[i]**2/(q_vec[i]**2 - s_vec[i])
        elif (q_vec[i]**2 >= s_vec[i] and alphas[i] >= inf):
            alphas[i] = s_vec[i]**2/(q_vec[i]**2 - s_vec[i])
        elif (q_vec[i]**2 <= s_vec[i] and alphas[i] <= inf):
            alphas[i] = inf
        elif (q_vec[i]**2 <= s_vec[i] and alphas[i] >= inf):
            pass

    beta_inv_num = np.linalg.norm(t_train - PHI @ mean)**2
    gamma_vec = np.zeros(M)
    for j in range(M):
        gamma_vec[j] = 1 - alphas[i]*Sigma[i][i]
    beta_inv_denum = N - np.sum(gamma_vec)
    beta_inv = beta_inv_num/beta_inv_denum
    beta = 1/beta_inv
#    print (iternum)
#    print (beta)
#    print (alphas)
    if abs(beta-beta_init) < 0.01 and abs(np.sum(alphas - alphas_init)) < 0.01:
        break
    else:
        alphas_init = alphas
        beta_init = beta
        
A = np.diag(alphas)
Sigma = get_Sigma(beta, A, PHI)
mean  = get_mean(beta, Sigma, PHI, t_train)
t_test = np.zeros(len(X_test))
for i, x_test in enumerate(X_test):
    t_test[i] = mean.T @ basis_function(x_test)

Util.plot_func(X_true, t_true, X_train, t_train, X_test, t_test)
