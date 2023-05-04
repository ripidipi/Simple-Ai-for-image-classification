import torch
import matplotlib.pyplot as plt
from torch import nn
from downloads.helper_functions import plot_decision_boundary
import numpy as np
import torchvision
from torchvision import datasets 
from torchvision import models 
from torchvision.transforms import ToTensor 
import torch.utils.data.dataset
import torch.utils.data.dataloader


train_data = datasets.FashionMNIST(
    root='data', # Where to download data
    train=True, # Do we want training dataset
    download=True, 
    transform=ToTensor(), 
    target_transform=None,
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(), 
    target_transform=None,
)

print(train_data, test_data)

# img = train_data[0][0]
# label = train_data[0][1]
# print(f"Image:\n {img}") 
# print(f"Label:\n {label}")


def plot_data() -> None:
    class_names = train_data.classes

    fig = plt.figure(figsize=(9, 9))
    rows, cols = 4, 4

    for i in range(1, rows*cols+1):
        random_ind = torch.randint(0, len(train_data), size=[1]).item()
        img, label = train_data[random_ind]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(class_names[label])
        plt.axis('off')
        plt.xlabel(False)

    plt.show()
    
# plot_data()









