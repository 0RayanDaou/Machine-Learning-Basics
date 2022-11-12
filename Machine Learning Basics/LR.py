import numpy as np


class LogisticRegression(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=0):
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        z = x.dot(self.w.T) + self.b
        f = 1 / (1 + np.exp(-z))
        return f.T

    def loss(self, x, y):
        n = x.shape[0]
        z = np.transpose(np.dot(x, np.transpose(self.w)) + self.b)
        f = (1 / n) * np.sum(np.log(1 + np.exp(-y * z))) + (1 / 2) * self.l2_reg * np.sum(np.transpose(self.w) * self.w)
        return f

    def grad_loss_wrt_b(self, x, y):
        y = y[:, np.newaxis]
        z = np.dot(x, np.transpose(self.w)) + self.b
        f = np.mean(-y / (1 + np.exp(y * z)), axis=0)
        return f

    def grad_loss_wrt_w(self, x, y):
        y = y[:, np.newaxis]
        z = np.dot(x, np.transpose(self.w)) + self.b
        f = np.mean(-(y * x) / (1 + np.exp(y * z)), axis=0)
        return f

    def fit(self, x, y):
        self.w = np.random.rand(1, x.shape[1])
        self.b = 0
        for i in range(0, self.n_epochs):
            self.b = self.b - self.lr * self.grad_loss_wrt_b(x, y)
            self.w = self.w - self.lr * self.grad_loss_wrt_w(x, y)

    def predict(self, x):
        y = np.array([], dtype=int)
        for point in x:
            f = self.forward(point)
            if f > 0.5:
                y = np.append(y, 1)
            else:
                y = np.append(y, -1)

        return y
