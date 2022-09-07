import numpy as np
import matplotlib.pyplot as plt
import stateml

def func(X):
    y = np.sin(X) + X

    return y

X_true = np.arange(-3, 4,0.01)
t_true = func(X_true)

noise = 0.1
X_train = np.arange(-3, 4, 1)
t_train = func(X_train) + noise * np.random.randn(*X_train.shape)

X_pred = np.arange(-3, 4,0.5)
t_pred = []
for x in X_pred:
    mu, cov = stateml.predict(X_train, x, t_train)
    t_pred.append(mu)
    
t_pred = np.array(t_pred)

plt.plot(X_true, t_true)
plt.scatter(X_train, t_train, c = 'r')
plt.plot(X_pred, t_pred, c = 'g')
plt.show()
