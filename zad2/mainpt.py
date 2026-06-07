import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]]).float()
T = torch.tensor([[0],[0],[0],[1]]).float()

model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

fs = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.1)

epochs = 100

for i in range(epochs):
    Y = model(X)
    loss = fs(Y,T)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

pred = model(X)
wynik = (pred > 0.5).int()
print(wynik)