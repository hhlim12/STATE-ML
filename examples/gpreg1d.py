import numpy as np
import matplotlib.pyplot as plt
from stateml import Kernel, GP, Util

def func(X):
    #y = np.sin(X) + X
    y = np.sin(3*X + 1) + X
    #y = np.sin(2*X**2+1) + X

    return y

X_true = np.arange(-3, 4,0.01)
t_true = func(X_true)

noise = 0.1
X_train = np.arange(-3, 4, 1)
t_train = func(X_train) + noise * np.random.randn(*X_train.shape)

X_test = np.arange(-3, 4, 0.2)

kernel = Kernel()
gp     = GP(X_train, t_train, kernel, noise) 
gp.optimize_params()
t_test, cov_tests = gp.predict_func(X_test)
nll = gp.get_nll()
mae = Util.calc_mae(X_test, t_test, func(X_test))
print ("sigma_f: ", gp.kernel.sigma_f)
print ("lkernel: ", gp.kernel.lscale)
print ("nll: ", nll)
print ("mae: ", mae)
Util.plot_func(X_true, t_true, X_train, t_train, X_test, t_test)
