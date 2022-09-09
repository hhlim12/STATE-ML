import numpy as np
from numpy.linalg import inv, det
from scipy.optimize import minimize

class Kernel:
    "Class for kernel"
    def __init__(self, sigma_f = 1, lscale = 1):
            self.sigma_f = sigma_f
            self.lscale  = lscale
    def __call__(self, x1, x2):
        sqdist = (x1-x2)**2
        k      = self.sigma_f**2 * np.exp(-0.5*sqdist / self.lscale**2)
        return k

class GP:
    "Class for GP"
    def __init__(self, X_train, t_train, kernel, noise):
        self.kernel  = kernel
        self.noise   = noise
        self.X_train = X_train
        self.t_train = t_train
        
    def calc_K(self):
        N = len(self.X_train)
        K = np.zeros((N, N))
        for i, x1 in enumerate(self.X_train):
            for j, x2 in enumerate(self.X_train):
                K[i,j] = self.kernel(x1, x2)

        self.K = K
        return K
    
    def predict_func(self, X_test):
        N = len(self.X_train)
        K = self.calc_K()
        self.X_test = X_test
        mu_tests, cov_tests = [], []
        for x_test in X_test:
            # Calc kvec
            kvec = np.zeros(N)
            for i, x_train in enumerate(X_train):
                kvec[i] = self.kernel(x_train, x_test)
        
            c = kernel(x_test, x_test) + self.noise
            C = K + self.noise*np.eye(N, N)
            
            self.C = C

            mu_tests.append(kvec.T @ inv(C) @ t_train)
            cov_tests.append(c - kvec.T @ inv(C) @ kvec)
            
        self.mu_tests = np.array(mu_tests)
        self.cov_tests = np.array(cov_tests)
        return (np.array(mu_tests), np.array(cov_tests))
        
    def get_nll(self, params=None):
        if params is None:
            params = [self.kernel.sigma_f,self.kernel.lscale]
        N    = len(self.X_train)
        self.kernel = Kernel(sigma_f = params[0], lscale = params[1])
        K    = self.calc_K()
        C    = K + self.noise*np.eye(N, N)
        nll  = 0.5*np.log(det(C)) + 0.5*t_train.T@inv(C)@t_train  + 0.5*N*np.log(2*np.pi)
        
        self.nll = nll
        
        return (nll)
    
    def optimize_params(self):
        opt_params = minimize(self.get_nll, x0 = [self.kernel.sigma_f, self.kernel.lscale])
        #, method='L-BFGS-B', options={'gtol': 1e-8})
        self.kernel.sigma_f = opt_params.x[0]
        self.kernel.lscale = opt_params.x[1]
        nll = self.get_nll(params=opt_params.x)
        self.nll = nll

class Util:
    def __init__():
        pass
    
    def plot_func(X_true, t_true, X_train, t_train, X_test, t_test):
        plt.plot(X_true, t_true, label='true')
        plt.scatter(X_train, t_train, c = 'r', label='train')
        plt.plot(X_test, t_test, c = 'g', label='predict')
        plt.legend()
        plt.show()
        
    def calc_mae(X_test, t_test):
        t_true = func(X_test)
        sqerror = np.abs(t_true-t_test)
        mae = np.mean(sqerror)
    
        return (mae)
