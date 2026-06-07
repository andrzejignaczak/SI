import numpy as np

class SLP:
    def __init__(self,n,m):
        self.W = np.random.randn(n,m)
        self.b = np.random.randn(1,m)

    def forward(self,X):
         return (X @ self.W + self.b > 0).astype(int)

    def backward(self,X,T):
        Y = self.forward(X)
        E = T-Y
        self.W = self.W + X.T @ E / X.shape[0]
        self.b = self.b + np.mean(E,axis=0)

    def train(self,X,T,epoch):
        for i in range(epoch):
            self.backward(X,T)
            
