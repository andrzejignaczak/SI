import numpy as np
import matplotlib.pyplot as plt
import os

class SLP:
    def __init__(self, n, m):
        self.W=np.random.randn(n,m)
        self.b=np.random.randn(m,1)
    def forward(self,X):
        return (X @ self.W + self.b.T > 0).astype(int)
    def backward(self,X,T):
        Y=self.forward(X)
        E=T-Y
        self.W=self.W + X.T @ E / X.shape[0]
        self.b=self.b + np.mean(E,axis=0).reshape(-1,1)
    def train(self,X,T,epoch):
        for i in range(epoch):
            self.backward(X,T)
X=np.array([[0,0],[0,1],[1,0],[1,1]])
T=np.array([[0],[0],[0],[1]])
P=SLP(2,1)

if os.path.exists('perceptron_weights.npz'):
    data=np.load('perceptron_weights.npz')
    P.W=data['W']
    P.b=data['b']
else:
    P.train(X,T,100)
    np.savez('perceptron_weights.npz',W=P.W,b=P.b)

print(P.forward(X))

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Granica decyzyjna Perceptronu')
for i in range(len(X)):
    if T[i]==0:
        plt.scatter(X[i][0],X[i][1],color='red',label='Klasa 0' if i==0 else "")
    else:        plt.scatter(X[i][0],X[i][1],color='blue',label='Klasa 1' if i==3 else "")
x1=np.linspace(-0.5,1.5,100)    
x2=-(P.W[0][0]*x1 + P.b[0][0])/P.W[1][0]
plt.plot(x1,x2,color='green',label='Granica decyzyjna')
plt.legend()
plt.grid()
plt.show()
    