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

class NetBin(nn.Module):
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
    
model = NetBin().to(device)

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

print(test_acc)
plot_pred(model)

def plot(x: torch.Tensor) -> None:
     plt.plot(x)
     plt.show()

# def Tanh(x) -> torch.Tensor:
#     return np.sinh(x) * np.cosh(x)

# A = torch.arange(-10, 10, 1., dtype=torch.float32).unsqueeze(dim=1)

# b = nn.Tanh()

# plot(b(A))

NUM_CLASSES = 5
NUM_FEATURES = 2
NUM_HIDENS = 16

N = 100 # number of points per class
D = 2 # dimensionality
K = NUM_CLASSES # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

def plot_spiral() -> None:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.show()

# plot_spiral()

X_spiral = torch.from_numpy(X).type(torch.float32)
y_spiral = torch.from_numpy(y).type(torch.LongTensor)

X_spiral_train, X_spiral_test, y_spiral_train, y_spiral_test = train_test_split(X_spiral,
                                                                                y_spiral,
                                                                                test_size=0.4)

X_spiral_train, y_spiral_train = X_spiral_train.to(device), y_spiral_train.to(device)
X_spiral_test, y_spiral_test = X_spiral_test.to(device), y_spiral_test.to(device)


class NetMulti(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=NUM_FEATURES, out_features=NUM_HIDENS),
            nn.Tanh(),
            nn.Linear(in_features=NUM_HIDENS, out_features=NUM_HIDENS),
            nn.ReLU(),
            nn.Linear(in_features=NUM_HIDENS, out_features=NUM_HIDENS),
            nn.Tanh(),
            nn.Linear(in_features=NUM_HIDENS, out_features=NUM_CLASSES),
        )
    
    def forward(self, x):
        return self.layers(x)
    
modelM = NetMulti().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(modelM.parameters(), 
                            lr=0.008)

acc_fn = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)

epochs = 1000

for epoch in range(epochs):
    
    modelM.train()

    y_logits = modelM(X_spiral_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, 
                   y_spiral_train)
    
    acc = acc_fn(y_pred, 
                 y_spiral_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()

    with torch.inference_mode():
        test_logits = modelM(X_spiral_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_spiral_test)

        test_acc = acc_fn(test_preds,
                           y_spiral_test)
        

def plot_predictions(model) -> None:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Train')
    plot_decision_boundary(model, X_spiral_train, y_spiral_train)
    plt.subplot(1, 2, 2)
    plt.title('Test')
    plot_decision_boundary(model, X_spiral_test, y_spiral_test)
    plt.show()

print(test_acc)
plot_predictions(modelM)

    








