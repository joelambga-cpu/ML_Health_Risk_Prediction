import torch
import torch.optim as optim
from .model import RiskNet

def train_torch(X_train, y_train):
    model = RiskNet(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    for epoch in range(30):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    return model
