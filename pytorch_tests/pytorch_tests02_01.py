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
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch import nn
import sys


NUM_CLASSES = 4
NUM_FEATURES = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 1. creat multi_class data

X_blob, y_blob = make_blobs(n_samples=1500,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.3)

# 2. turn data into torch tensor

X_blob = torch.tensor(X_blob, dtype=torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.long)

# 3. split data into train and test

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, 
                                                                        test_size=0.4,)

def blob_plot(x, y) -> None:
    plt.figure(figsize=(10, 7))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()
    
# blob_plot(X_blob, y_blob)

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=16) -> None:
        super().__init__()
        """Initializes all required hyperparameters for a multi-class classification model.

        Args:
            input_features (int): Number of input features to the model.
            out_features (int): Number of output features of the model
              (how many classes there are).
            hidden_units (int): Number of hidden units between layers, default 8.
        """
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
        

model_3 = BlobModel(input_features=NUM_FEATURES,
                    output_features=NUM_CLASSES,).to(device)
                
#print(model_3)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model_3.parameters(), 
                             lr=0.001)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

model_3.eval()
with torch.inference_mode():
    y_logits = model_3(X_blob_test)

y_pred_probs = torch.softmax(y_logits, dim=1)
y_pred = torch.argmax(y_pred_probs, dim=1)

# print(y_pred)

epochs = 500

for epoch in range(epochs):

    model_3.train()

    y_logits = model_3(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                              y_pred=test_preds)
        

    # if epoch % 10 == 0:
    #     print(f"Epoch: {epoch}||| loss: {loss:.5f}\n Acc: {acc:.3f}%||| test loss: {test_loss:.5f}\ntest acc: {test_acc:.3f}%")


def plot_predictions(model) -> None:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_blob_train, y_blob_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_blob_test, y_blob_test)
    plt.show()

plot_predictions(model_3)