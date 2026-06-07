import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

X = np.array([[0,0],[0,1],[1,0],[1,1]])
T = np.array([[0],[0],[0],[1]])

nn = Sequential([Input((2,)), Dense(1, activation='sigmoid')])

nn.compile(optimizer=Adam(learning_rate=.1), loss='mse')

nn.fit(X, T, epochs=100, verbose=2)

pred = nn.predict(X)
wynik = (pred > 0.5).astype(int)
print(wynik)