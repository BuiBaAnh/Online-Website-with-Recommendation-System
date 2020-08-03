from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from math import isnan
class MF(object):
    def __init__(self, Y, K, lam = 0.1, Xinit = None, Winit = None,
    learning_rate = 0.5, max_iter = 1000, print_every = 100):
        self.Y = Y # represents the utility matrix
        self.K = K #
        self.lam = lam # regularization parameter
        self.learning_rate = learning_rate # for gradient descent
        self.max_iter = max_iter # maximum number of iterations
        self.print_every = print_every # print loss after each a few iters
        self.n_users = int(np.max(Y[:, 0])) + 1
        self.n_items = int(np.max(Y[:, 1])) + 1
        self.n_ratings = Y.shape[0] # number of known ratings
        self.X = np.random.randn(self.n_items, K) if Xinit is None else Xinit
        self.W = np.random.randn(K, self.n_users) if Winit is None else Winit
        self.b = np.random.randn(self.n_items) # item biases
        self.d = np.random.randn(self.n_users) # user biases

    def loss(self):
        L = 0
        for i in range(self.n_ratings):
        # user_id, item_id, rating
            n, m, rating = int(self.Y[i, 0]), int(self.Y[i, 1]), self.Y[i, 2]
            L += 0.5*(self.X[m].dot(self.W[:, n]) + self.b[m] + self.d[n] - rating)**2
        L /= self.n_ratings
        # regularization, don't ever forget this
        return L + 0.5*self.lam*(np.sum(self.X**2) + np.sum(self.W**2))
    def updateXb(self):
        for m in range(self.n_items):
            # get all users who rated item m and get the corresponding ratings
            ids = np.where(self.Y[:, 1] == m)[0] # row indices of items m
            user_ids, ratings = self.Y[ids, 0].astype(np.int64), self.Y[ids, 2]
            Wm, dm = self.W[:, user_ids], self.d[user_ids]
            for i in range(30): # 30 iteration for each sub problem
                xm = self.X[m]
                error = xm.dot(Wm) + self.b[m] + dm - ratings
                grad_xm = error.dot(Wm.T)/self.n_ratings + self.lam*xm
                grad_bm = np.sum(error)/self.n_ratings
                # gradient descent
                self.X[m] -= self.learning_rate*grad_xm.reshape(-1)
                self.b[m] -= self.learning_rate*grad_bm
    def updateWd(self): # and d
        for n in range(self.n_users):
        # get all items rated by user n, and the corresponding ratings
            ids = np.where(self.Y[:,0] == n)[0] # row indices of items rated by user n
            item_ids, ratings = self.Y[ids, 1].astype(np.int64), self.Y[ids, 2]
            Xn, bn = self.X[item_ids], self.b[item_ids]
            for i in range(30): # 30 iteration for each sub problem
                wn = self.W[:, n]
                error = Xn.dot(wn) + bn + self.d[n] - ratings
                grad_wn = Xn.T.dot(error)/self.n_ratings + self.lam*wn
                grad_dn = np.sum(error)/self.n_ratings
                # gradient descent
                self.W[:, n] -= self.learning_rate*grad_wn.reshape(-1)
                self.d[n] -= self.learning_rate*grad_dn
    def fit(self):
        for it in range(self.max_iter):
            self.updateWd()
            self.updateXb()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y)
                print('iter = %d, loss = %.4f, RMSE train = %.4f'%(it + 1,
                self.loss(), rmse_train))  
    def pred(self, u, i):
        u, i = int(u), int(i)
        pred = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u]# + bias
        return max(0, min(5, pred)) # pred should be between 0 and 5 in MoviesLen
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0] # number of test
        SE = 0 # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2
        RMSE = np.sqrt(SE/n_tests)
        return RMSE
    def find(self,user):
        _ = []
        u = int(user)
        for i in range(self.n_items):
            pred = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u]# + bias
            _.append((i,pred))
        return _
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols)
print(ratings_test.shape)
rate_train = ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()
# indices start from 0
print(type(rate_train[2][2]))
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1
rs = MF(rate_train, K = 50, lam = .01, print_every = 5, learning_rate = 10,
max_iter = 50)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print('\nMatrix Factorization CF, RMSE = %.4f' %RMSE)
print(rs.find(5))
