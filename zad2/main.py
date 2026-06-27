import numpy as np
import perceptron as nn

X = np.array([[0,0],[0,1],[1,0],[1,1]])
T = np.array([[0],[0],[0],[1]])

P = nn.SLP(X.shape[1],T.shape[1])
P.train(X,T,50)
print(P.forward(X))
