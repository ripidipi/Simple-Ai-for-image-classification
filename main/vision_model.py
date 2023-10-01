import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import pathlib
from dataset_setting import class_names

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using : {device}")

class Food(nn.Module):
    "Model architecture"
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Softmax2d(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Softmax2d(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=4,
                      stride=1,
                      padding=1),
            nn.Softmax2d(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*27*27,
                      out_features=output_shape)
        )

    def forward(self, x):
        # print(x.shape)
        # x = self.conv_block_1(x)
        # print(x.shape)
        # x = self.conv_block_2(x)
        # print(x.shape)
        # x = self.conv_block_3(x)
        # print(x.shape)
        # x = self.classifier(x)
        # return x
        x = x.to(device)
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))

model = Food(input_shape=3,
               hidden_units=16,
               output_shape=len(class_names)).to(device)
