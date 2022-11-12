import math
import numpy as np
from scipy import stats

class KNearestNeighbors(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        y = np.array([], dtype=int)
        euc_dist_mat = []
        for point in x:
            euc_dist_list = []
            for ind in range(len(x)):
                euc_dist = 0
                dist_sq_x = pow(point[0] - self.x_train[ind][0], 2)
                dist_sq_y = pow(point[1] - self.x_train[ind][1], 2)
                dist_sq = dist_sq_x + dist_sq_y
                euc_dist = math.sqrt(dist_sq)
                euc_dist_list.append(euc_dist)
            euc_dist_mat.append(np.argsort(euc_dist_list))
        euc_dist_mat = np.array(euc_dist_mat)
        for distance in euc_dist_mat:
            yneg1 = 0
            ypos1 = 0
            for ind in distance[0:self.k]:
                if self.y_train[ind] == 1:
                    ypos1 += 1
                else:
                    yneg1 += 1
            if ypos1 > yneg1:
                y = np.append(y, 1)
            else:
                y = np.append(y, -1)
        return y