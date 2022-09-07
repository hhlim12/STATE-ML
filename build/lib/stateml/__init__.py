import numpy as np
from numpy.linalg import inv

def kernel(x1, x2, sigma_f = 1, l = 1):
    sqdist = (x1-x2)**2
    k      = sigma_f**2 * np.exp(-0.5*sqdist / l**2)
    return k

def get_K(X_train):
    K = np.zeros((len(X_train), len(X_train)))
    for i, x1 in enumerate(X_train):
        for j, x2 in enumerate(X_train):
            K[i,j] = kernel(x1, x2)

    return K

def get_kvec(X_train, x_test):
    kvec = np.zeros(len(X_train))
    for	i, x1 in enumerate(X_train):
        kvec[i] = kernel(x1, x_test)

    return kvec

def get_c(x_test):
    c =	kernel(x_test, x_test)

    return c

def predict(X_train, x_test, t_train, beta=0.1):
    N    = len(X_train)
    K    =	get_K(X_train)
    kvec = get_kvec(X_train, x_test)
    c    = get_c(x_test)
    C    = K + beta*np.eye(N, N)

    mu_test = kvec.T @ inv(C) @ t_train
    cov_test = c - kvec.T @ inv(C) @ kvec

    return (mu_test, cov_test)
