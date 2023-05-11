import torch
import matplotlib.pyplot as plt
from torch import nn
from downloads.helper_functions import accuracy_fn
import numpy as np
import torchvision
from torchvision import datasets 
from torchvision import models 
from torchvision.transforms import ToTensor 
import torch.utils.data.dataset
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)

''' CNN '''

class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=output_shape,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        print(x.shape)
        x = self.conv_block_2(x)
        print(x.shape)
        x = self.classifier(x)
        return x

model = FashionMNISTModelV2(input_shape=1,
                            output_shape=10,
                            hidden_units=10,).to(device)


rand_image_tensor = torch.randn(size=(1, 28, 28)).to(device)

model(rand_image_tensor.unsqueeze(0))





