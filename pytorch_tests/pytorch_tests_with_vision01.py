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
print(device)

BATCH_SIZE = 32

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

train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH_SIZE,
                              shuffle = True)

test_dataloader = DataLoader(dataset = test_data,
                             batch_size = BATCH_SIZE,
                             shuffle = False)

train_features_batch, train_labels_batch  = next(iter(train_dataloader))
class_names = train_data.classes

flatten_model = nn.Flatten()

x = train_features_batch[0]

output = flatten_model(x)

class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int ):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer_stack(x)
    

model_0 = FashionMNISTModelV1(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

accuracy_fn = accuracy_fn

def print_train_time(start: float, 
                     end: float,
                     device: torch.device = None ):
    total_time = end - start
    print(f'Train time on {device} :  {total_time:.5f} seconds')
    return total_time


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """ Performs a training with model trying to learn on data_loader. """
    train_loss, train_acc = 0, 0

    model.train()

    for batch, (img_train, label) in enumerate(data_loader):
        
        img_train, label = img_train.to(device), label.to(device)

        model.eval()

        # forward pass
        y_pred = model(img_train)

        # loss calculation
        loss = loss_fn(y_pred, label)
        train_loss += loss

        train_acc += accuracy_fn(y_true=label,
                                 y_pred=y_pred.argmax(dim=1))
    
        # zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer
        optimizer.step()

    
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f' Train loss: {train_loss:.4f} \nTrain accuracy: {train_acc:.2f}')


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    """ Performs a testing loop step on model going over data_loader. """

    test_loss, test_acc = 0, 0
    
    model.eval()

    with torch.inference_mode():
        for img_test, lable_test in data_loader:
            
            img_test, lable_test = img_test.to(device), lable_test.to(device)

            test_pred = model(img_test)
            test_loss += loss_fn(test_pred, lable_test)
            test_acc += accuracy_fn(y_true=lable_test, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc  /= len(test_dataloader)
        
        print(f'Test loss: {test_loss:.4f} \nTest accuracy: {test_acc:.2f}')



start_time = timer()

epochs = 10

for epoch in tqdm(range(epochs)):
    
    train_step(model=model_0,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               )
    
    test_step(model=model_0,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              )





end_time = timer()
print_train_time(start=start_time, end=end_time, device=device)            