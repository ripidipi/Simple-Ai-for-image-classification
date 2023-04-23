import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from downloads.helper_functions import plot_decision_boundary
from torchmetrics import Accuracy
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)


X_moon, y_moon = make_moons(n_samples=2000,
                            noise=0.1)

def moon_plot():
    plt.scatter(X_moon[:, 0], X_moon[:, 1], c=y_moon)
    plt.show()

# moon_plot()
X_moon = torch.from_numpy(X_moon).type(torch.float32)
y_moon = torch.from_numpy(y_moon).type(torch.float32)


X_train, X_test, y_train, y_test = train_test_split(X_moon,
                                                    y_moon,
                                                    test_size=0.4)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

def Tanh(x) -> torch.Tensor:
    return np.sinh(x) * np.cosh(x)

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, x):
        return self.layers(x)
    
model = Net().to(device)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=0.01)

epochs = 300

for epoch in range(epochs):

    model.train()

    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, 
                   y_train)
    
    optimizer.zero_grad()

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    loss.backward()

    optimizer.step()

    model.eval()

    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(y_logits))
        
        test_acc = accuracy_fn(y_true=y_train,
                                y_pred=y_pred)
        
        test_loss = loss_fn(test_logits,
                             y_test)
        
    # if epoch % 10 == 0:
    #     print(f"Epoch: {epoch}||| loss: {loss:.5f}\n Acc: {acc:.3f}%||| test loss: {test_loss:.5f}\ntest acc: {test_acc:.3f}%")

        

def plot_pred(model):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Train')
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title('Test')
    plot_decision_boundary(model, X_test, y_test)
    plt.show()

plot_pred(model)

def plot(x: torch.Tensor) -> None:
    plt.plot(x)
    plt.show()

def Tanh(x) -> torch.Tensor:
    return np.sinh(x) * np.cosh(x)

# A = torch.arange(-10, 10, 1., dtype=torch.float32).unsqueeze(dim=1)

# plot(Tanh(A))