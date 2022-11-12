import pickle

from KNN import KNearestNeighbors
from LR import LogisticRegression
from LSF import LeastSquares
import datasets
import numpy as np
import matplotlib.pyplot as plt


class RunAlgorithms:
    def __init__(self):
        self.model = None
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None

    def initialize_data(self, ls=False):
        if ls:
            # least squares on a polynomial
            data = pickle.load(open("ls_data.pkl", "rb"))
            self.xtrain = data['x_train']
            self.ytrain = data['y_train']
            self.xtest = data['x_test']
            self.ytest = data['y_test']
        else:
            self.xtrain, self.ytrain, self.xtest, self.ytest = datasets.gaussian_dataset(n_train=800, n_test=800)

    def run_knn(self):
        kvalues = np.arange(1, 52, step=5)
        means = []
        for k in kvalues:
            self.model = KNearestNeighbors(k)
            self.model.fit(self.xtrain, self.ytrain)
            y_predicted = self.model.predict(self.xtest)
            mean = np.mean(y_predicted == self.ytest)
            means.append(mean)
        plt.plot(kvalues, means, label='Classification Test Accuracy')
        plt.xlabel('K values')
        plt.ylabel('Accuracy')
        plt.title('KNN Accuracy')
        plt.legend()
        plt.savefig('Classification Test Accuracy KNN')
        plt.show()

    def run_lr(self):
        lr_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 1, 5, 10])
        means = []
        for lr in lr_values:
            self.model = LogisticRegression()
            self.model.lr = lr
            self.model.fit(self.xtrain, self.ytrain)
            y_predicted = self.model.predict(self.xtest)
            mean = np.mean(y_predicted == self.ytest)
            means.append(mean)
        plt.plot(lr_values, means, label='Classification Test Accuracy')
        plt.xlabel('Learning Rates')
        plt.ylabel('Accuracy')
        plt.title('Logistic Regression Accuracy')
        plt.legend()
        plt.savefig('Classification Test Accuracy Logistic Regression')
        plt.show()

    def run_LSF(self):
        poly_degree = np.arange(1, 20, step=1)
        MSE_poly_test = []
        MSE_poly_train = []
        for i in poly_degree:
            self.model = LeastSquares(i)
            self.model.fit(self.xtrain, self.ytrain)
            y_predicted = self.model.predict(self.xtest)
            y_predicted_train = self.model.predict(self.xtrain)
            MSE_poly_test.append(self.MSE(self.ytest, y_predicted))
            MSE_poly_train.append(self.MSE(self.ytrain, y_predicted_train))
        plt.plot(poly_degree, MSE_poly_test, label='Test Error')
        plt.plot(poly_degree, MSE_poly_train, label='Train Error')
        plt.xlabel('Degree of Polynomial')
        plt.ylabel('Error')
        plt.title('Least Square Fitting Test/Train Error')
        plt.legend()
        plt.savefig('Classification Test Accuracy Least Square Fitting')
        plt.show()

    def MSE(self, actual, pred):
        return np.square(np.subtract(actual, pred)).mean()


# run1 = RunAlgorithms()
# run1.initialize_data(ls=False)
# run1.run_knn()
# run2 = RunAlgorithms()
# run2.initialize_data(ls=False)
# run2.run_lr()
# run3 = RunAlgorithms()
# run3.initialize_data(ls=True)
# run3.run_LSF()
