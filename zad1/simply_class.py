

import numpy as np
import scipy.io as sio

# Funkcje aktywacji
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

# Funkcja straty
def cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Definicja sieci MLP
class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, lr=0.01):
        self.W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2 / n_inputs)
        self.b1 = np.zeros((1, n_hidden))
        #self.b1 = np.random.randn(1, n_hidden)

        self.W2 = np.random.randn(n_hidden, n_outputs) * np.sqrt(2 / n_hidden)
        self.b2 = np.zeros((1, n_outputs))
        #self.b2 = np.random.rand(1, n_outputs)
        self.lr = lr

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2


    def backward(self, X, y_true):
        m = X.shape[0]

        # Gradient: softmax + cross-entropy
        dZ2 = self.A2 - y_true
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Propagacja wsteczna
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Aktualizacja wag
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y_true, epochs=100):
        for e in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy(y_true, y_pred)
            self.backward(X, y_true)
            if e % 10 == 0:
                print(f"Epoka {e}, loss = {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


# Wczytaj dane z pliku
data = sio.loadmat("danelitery.mat")
X = data["X"].T
T = data["T"].T

# Tworzenie sieci
model = MLP(n_inputs=35, n_hidden=60, n_outputs=26, lr=0.01)

# Uczenie
model.train(X, T, epochs=150)

# Predykcja
y_true = np.argmax(T, axis=1)
y_pred = model.predict(X)

# Test
accuracy = np.mean(y_true == y_pred)
print(f" Dokładność: {accuracy * 100:.2f}%")

# Przykład działania na kilku literach
for i in range(10):
    print(f"Test {i}: litera = {chr(65 + y_true[i])}, predykcja = {chr(65 + y_pred[i])}")


