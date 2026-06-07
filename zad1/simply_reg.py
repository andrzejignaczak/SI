

import numpy as np
import pandas as pd

# Funkcje aktywacji
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Błąd średniokwadratowy - funkcja straty
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

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
        self.A2 = self.Z2 # liniowa funkcja aktywacji
        return self.A2


    def backward(self, X, y_true):
        m = X.shape[0]

        # Gradient: regresja
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
            loss = mse(y_true, y_pred)
            self.backward(X, y_true)
            if e % 10 == 0:
                print(f"Epoka {e}, loss = {loss:.4f}")

    def predict(self, X):
        return self.forward(X)


# Wczytanie danych z pliku
data = pd.read_csv("Boston.csv")

# X = wszystkie kolumny oprócz "medv"
X = data.drop(columns=["medv"]).values
# T = kolumna docelowa "medv"
T = data["medv"].values.reshape(-1, 1)

# Normalizacja zmiennych
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Tworzenie sieci
model = MLP(n_inputs=X.shape[1], n_hidden=40,n_outputs=1, lr=0.01)

# Uczenie
model.train(X, T, epochs=150)

# Predykcja
y_pred = model.predict(X)

# Dopasowanie
print("\nMSE:", mse(T, y_pred))

# Przykładowe wartości na wyjściu i ich predykcji
for i in range(5):
    print(f"Rzeczywista cena: {T[i][0]:.1f}  |  Predykcja: {y_pred[i][0]:.1f}")


