import pickle

import numpy as np


class LeastSquares(object):
    def __init__(self, k):
        self.k = k
        self.coeff = None

    def fit(self, x, y):
        A = np.vstack([np.ones(len(x)), x])
        for i in range(2, self.k+1):
            A = np.vstack([A, x**i])
        A = A.T
        A_PI = np.linalg.pinv(A)
        self.coeff = A_PI.dot(y)


    def predict(self, x):
        A = np.vstack([np.ones(len(x)), x])
        for i in range(2, self.k + 1):
            A = np.vstack([A, x ** i])
        A = A.T
        return A.dot(self.coeff)