import time 
start_time = time.time()
import torch
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import pytorch_save_and_load_test
from downloads.helper_functions import plot_decision_boundary
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch import nn
import sys


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

""" Classification """

n_samples = 1000

x, y = make_circles(n_samples, 
                    noise=0.05,)

# print(x)
# print('x:\n', x[:5], '\ny:\n', y[:5])
# circles = pd.DataFrame({'X1': x[:, 0], 
#                         "X2": x[:, 1],
#                         "lable": y})

#print(circles.head(20))
def circl_plot(x_coord=x[:, 0], 
               y_coord=x[:, 1],
               dot_category=y):
    plt.scatter(x=x_coord,
            y=y_coord,
            c=dot_category,
            cmap=plt.cm.RdYlBu);
    plt.legend(prop={'size': 10});
    plt.show()

def plot(x: torch.Tensor) -> None:
    plt.plot(x)
    plt.show()

# circl_plot()

# turn data into tensors

X = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.4 # procent of data will be test
                                                    )


class CircleModelV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1) 
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x))


class CircleModelV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=8)
        self.layer_2 = nn.Linear(in_features=8, out_features=8)
        self.layer_3 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))


class CircleModelV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=16)
        self.layer_2 = nn.Linear(in_features=16, out_features=16)
        self.layer_3 = nn.Linear(in_features=16, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_0 = CircleModelV0().to(device)
model_1 = CircleModelV1().to(device)
model_2 = CircleModelV2().to(device)

# model_0 = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features=5, out_features=1)   
# ).to(device)

# with torch.inference_mode():
#     untrained_preds = model_0(X_test.to(device))
# print(len(untrained_preds), untrained_preds.shape)
# print(untrained_preds[:10])
# print(y_test[:10])


loss_fn = nn.BCEWithLogitsLoss()

optimizer_0 = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.001)

optimizer_1 = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)

optimizer_2 = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.07)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

# with torch.inference_mode():
#     y_logits = model_0(X_test.to(device))
# print(y_logits)

# y_pred = torch.sigmoid(y_logits)
# y_pred_lable = torch.round(torch.sigmoid(model_0(X_test.to(device))))
# print(torch.round(y_pred))
# print(y_test[:10])
#print(torch.eq(y_pred.squeeze(), y_pred_lable.squeeze()))

#print(y_pred.squeeze())

epochs = 2000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):

    model_2.train()

    y_logits = model_2(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,
                   y_train)
    
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred )
    
    optimizer_2.zero_grad()
    # optimizer_0.zero_grad()

    loss.backward()

    # optimizer_0.step()
    optimizer_2.step()

    model_2.eval()
    # model_0.eval()
    with torch.inference_mode():
        test_logits = model_2(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,
                            y_test)
        
        test_acc = accuracy_fn(y_true=y_train,
                               y_pred=y_pred)
        
    # if epoch % 100 == 0:
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


print(f'acc: {acc:.2f}')
plot_pred(model_2)



weight = 0.7
bias = 0.4
start = 0
end = 1
step = 0.01

X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

train_split = int(0.6 * len(X_regression))
X_train_reg, y_train_reg = X_regression[:train_split], y_regression[:train_split]
X_test_reg, y_test_reg = X_regression[train_split:], y_regression[train_split:]

model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)

# Set the number of epochs
epochs = 1

# Put data to target device
X_train_regression, y_train_regression = X_train_reg.to(device), y_train_reg.to(device)
X_test_regression, y_test_regression = X_test_reg.to(device), y_test_reg.to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.001)

for epoch in range(epochs):
    
    y_pred = model_2(X_train_regression)
    
    loss = loss_fn(y_pred, y_train_regression)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    
    model_2.eval()
    with torch.inference_mode():
      
      test_pred = model_2(X_test_regression)
       
      test_loss = loss_fn(test_pred, y_test_regression)
     
    # if epoch % 100 == 0: 
    #     print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")



# def plot_predictions(train_data=X_train,
# 					train_lables=y_train,
# 					test_data=X_test,
# 					test_lables=y_test,
# 					predictions=None):

	
# 	#Plot training data, test data and compares prediction
	
# 	plt.figure(figsize=(10, 7))
# 	#scatter - разброс
# 	#plot training data in blue
# 	plt.scatter(train_data, train_lables, c="b", s=4, label='training data')
# 	#plot testing data in red
# 	plt.scatter(test_data, test_lables, c="r", s=4, label='tasting data')
	
# 	#are there predictions?
# 	if predictions is not None:
# 		#if predictions is exist plot it 
# 		plt.scatter(test_data, predictions, c="g", s=5, label='Predictions')

# 	plt.legend(prop={'size': 10});
# 	plt.show()

# plot_predictions()

# a = torch.arange(-10, 10, 1., dtype=torch.float32).unsqueeze(dim=1)

# def sigmoid(x) -> torch.Tensor:
#     return 1 / (1 + np.exp(-x))

# plot(sigmoid(a))





















print(time.time() - start_time)

