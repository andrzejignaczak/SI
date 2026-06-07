import numpy as np
import matplotlib.pyplot as plt

class SLP:
    def __init__(self, n, m):
        self.W = np.random.randn(n, m)
        self.b = np.random.randn(1, m)
        
    def forward(self, X):
        return (X @ self.W + self.b > 0).astype(int)
    
    def backward(self, X, T):
        Y = self.forward(X)
        E = T - Y
        self.b = self.b + np.mean(E, axis=0)
        self.W = self.W + X.T @ E / X.shape[0]
        return np.all(E == 0)
        
    def train(self, X, T, epoch):
        for i in range(epoch):
            converged = self.backward(X, T)
            if converged:
                print(f"Converged at epoch {i}")
                break
            
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [0], [0], [1]])

Perceptron = SLP(2, 1)  # n wejść, m wyjść
Perceptron.train(X, T, 100)  # epok
print(Perceptron.forward(X))
print(Perceptron.W)
print(Perceptron.b)

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Granica decyzyjna Perceptronu')

for i in range(len(X)): #rysowanie punktów danych
    if T[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red', marker='o', s=1000, label='(0)' if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='blue', marker='x', s=1000, label='(1)' if i == 3 else "")
x_values = np.array([-0.5, 1.5]) 
y_values = -(Perceptron.W[0][0] * x_values + Perceptron.b[0][0]) / Perceptron.W[1][0]
plt.plot(x_values, y_values, color='green', linewidth=2, label='Granica decyzji (Plot)')
plt.xlim(-0.2, 1.2) #granica po osi X
plt.ylim(-0.2, 1.2) #granica po osi Y
plt.legend() #wyswietlanie legendy
plt.grid(True, linestyle='--', alpha=0.5) 
plt.show()