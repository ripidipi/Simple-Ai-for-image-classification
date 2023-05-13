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
# print(train_data, test_data)

# img = train_data[0][0]
# label = train_data[0][1]
# print(f"Image:\n {img}") 
# print(f"Label:\n {label}")


def plot_data0() -> None:
    
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
    
# plot_data0()

def plot_data1():
    random_ind = torch.randint(0, len(train_features_batch), size=[1]).item()
    img, labels = train_features_batch[random_ind], train_labels_batch[random_ind]
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(class_names[labels])
    plt.axis('off')
    plt.xlabel(False)

    plt.show()

# plot_data1()


flatten_model = nn.Flatten()

x = train_features_batch[0]

output = flatten_model(x)

# print(f'Shape befor: {x.shape}')
# print(f'Shape after: {output.shape}')


class FashionMNISTModelV0(nn.Module):
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
    
model_0 = FashionMNISTModelV0(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names),
).to(device)

# dummy_x = torch.rand(1, 1, 28, 28).to(device)

# print(model_0(dummy_x).shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)
 
def print_train_time(start: float, 
                     end: float,
                     device: torch.device = None ):
    total_time = end - start
    print(f'Train time on {device} :  {total_time:.5f} seconds')
    return total_time

start_time = timer()

epochs = 5

for epoch in tqdm(range(epochs)):
    
    train_loss = 0

    for batch, (img, label) in enumerate(train_dataloader):
        
        img, label = img.to(device), label.to(device)

        model_0.eval()

        # forward pass
        y_pred = model_0(img)

        # loss calculation
        loss = loss_fn(y_pred, label)
        train_loss += loss

        # zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer
        optimizer.step()

        if batch % 400 == 0:
            print(f'looked {batch * len(img)} / {len(train_dataloader.dataset)}')
    
    # divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader)
    test_acc, test_loss = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for img_test, lable_test in test_dataloader:
            
            img_test, lable_test = img_test.to(device), lable_test.to(device)

            test_pred = model_0(img_test)
            test_loss += loss_fn(test_pred, lable_test)
            test_acc += accuracy_fn(y_true=lable_test, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)

        test_acc  /= len(test_dataloader)


print(f'train loss: {train_loss:.3f} \ntest loss: {test_loss:.3f} \n test acc: {test_acc:.3f}')




end_time = timer()
print_train_time(start=start_time, end=end_time, device=device)



def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for img, label in data_loader:

            img, label = img.to(device), label.to(device)

            y_pred = model(img)

            loss += loss_fn(y_pred, label)
            acc += accuracy_fn(y_pred = y_pred.argmax(dim=1),
                               y_true = label)
            
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {'model name:': model.__class__.__name__, 
            'model_loss': loss.item(),
            'model acc': acc}


model_0_res = eval_model(model=model_0,
                         data_loader=test_dataloader,
                         loss_fn=loss_fn,
                         accuracy_fn=accuracy_fn,
                         )


print(model_0_res)






