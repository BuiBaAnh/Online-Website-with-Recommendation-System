from __future__ import print_function
import pandas as pd
import numpy as np
from math import isnan
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class MF(object):
    def __init__(self, Y, K, reg = 0.1, Xinit = None, Winit = None,
    learning_rate = 0.5, max_iter = 1000, print_loss = 100):
        self.Y = Y # Utility Matrix
        self.K = K 
        self.reg = reg # regularization
        self.learning_rate = learning_rate # learning rate for gradient descent
        self.max_iter = max_iter # number of loop
        self.print_loss = print_loss # function print loss
        self.known_users = int(np.max(Y[:, 0])) + 1
        self.known_item = int(np.max(Y[:, 1])) + 1
        self.known_rating = Y.shape[0] # known ratings
        # initial random value for each matrix
        self.X = np.random.randn(self.known_item, K) if Xinit is None else Xinit
        self.W = np.random.randn(K, self.known_users) if Winit is None else Winit
        self.b = np.random.randn(self.known_item) # item biases
        self.d = np.random.randn(self.known_users) # user biases

    def loss(self):
        L = 0
        for i in range(self.known_rating):
        # get known user_id, item_id, rating
            n, m, rating = int(self.Y[i, 0]), int(self.Y[i, 1]), self.Y[i, 2]
            L += 0.5*(self.X[m].dot(self.W[:, n]) + self.b[m] + self.d[n] - rating)**2
        L /= self.known_rating
        # regularization
        return L + 0.5*self.reg*(np.sum(self.X**2) + np.sum(self.W**2))

    def updateX(self):
        for m in range(self.known_item):
            # get all users who rate item m
            ids = np.where(self.Y[:, 1] == m)[0] 
            user_ids, ratings = self.Y[ids, 0].astype(np.int64), self.Y[ids, 2]
            Wm, dm = self.W[:, user_ids], self.d[user_ids]
            for i in range(30): # using 30 loops
                xm = self.X[m]
                error = xm.dot(Wm) + self.b[m] + dm - ratings
                grad_xm = error.dot(Wm.T)/self.known_rating + self.reg*xm
                grad_bm = np.sum(error)/self.known_rating
                # gradient descent
                self.X[m] -= self.learning_rate*grad_xm.reshape(-1)
                self.b[m] -= self.learning_rate*grad_bm

    def updateW(self): 
        for n in range(self.known_users):
        # get all items rated by user n
            ids = np.where(self.Y[:,0] == n)[0] 
            item_ids, ratings = self.Y[ids, 1].astype(np.int64), self.Y[ids, 2]
            Xn, bn = self.X[item_ids], self.b[item_ids]
            for i in range(30): # using 30 loops
                wn = self.W[:, n]
                error = Xn.dot(wn) + bn + self.d[n] - ratings
                grad_wn = Xn.T.dot(error)/self.known_rating + self.reg*wn
                grad_dn = np.sum(error)/self.known_rating
                # gradient descent
                self.W[:, n] -= self.learning_rate*grad_wn.reshape(-1)
                self.d[n] -= self.learning_rate*grad_dn

    def fit(self):
        for it in range(self.max_iter):
            self.updateW()
            self.updateX()
            if (it + 1) % self.print_loss == 0:
                rmse = self.loss_RMSE(self.Y)
                print('iter = %d, loss = %.4f, RMSE train = %.4f'%(it + 1, self.loss(), rmse))  

    def predict(self, u, i):
        u, i = int(u), int(i)
        predict = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u]
        return max(0, min(5, predict)) #rating in range(0-5)

    def loss_RMSE(self, rate_test):
        n_tests = rate_test.shape[0] 
        SE = 0 
        for n in range(n_tests):
            predict = self.predict(rate_test[n, 0], rate_test[n, 1])
            SE += (predict - rate_test[n, 2])**2
        RMSE = np.sqrt(SE/n_tests)
        return RMSE

    def find(self,user):
        _ = []
        u = int(user)
        for i in range(self.known_item):
            predict = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u]
            _.append((i,predict))
        return _

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols)
print(ratings_test.shape)
rate_train = ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1
rs = MF(rate_train, K = 50, reg = .01, print_loss = 5, learning_rate = 10, max_iter = 50)
rs.fit()

# loss on test data
RMSE = rs.loss_RMSE(rate_test)
print('RMSE = %.4f' %RMSE)
print(rs.find(5))
