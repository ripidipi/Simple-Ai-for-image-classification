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
from pytorch_tests_with_vision00_01 import *
from pathlib import Path
import random 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# create model save path
MODEL_NAME = "vision_model.pth"
MODEL_SAVING_PATH = (f"{MODEL_PATH}/{MODEL_NAME}")

def save(model):    
    print(f"Saving path for model is {MODEL_SAVING_PATH}")
    # save the models state dict
    torch.save(obj=model.state_dict(),   
                f=MODEL_SAVING_PATH)


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
                      out_channels=hidden_units,
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
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        # return self.classifier(self.conv_block_2(self.conv_block_1(x)))
        return x

model = FashionMNISTModelV2(input_shape=1,
                            output_shape=10,
                            hidden_units=128,).to(device)


rand_image_tensor = torch.randn(size=(1, 28, 28)).to(device)

# print(model(rand_image_tensor.unsqueeze(0).to(device)))

# Setup a loss and optimizer

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.001)


start_time = timer()

epochs = 0

for epoch in tqdm(range(epochs)):
    
    train_step(model=model,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               )
    
    test_step(model=model,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              )


end_time = timer()
print_train_time(start=start_time, end=end_time, device=device) 


model_res = eval_model(model=model,
                         data_loader=test_dataloader,
                         loss_fn=loss_fn,
                         accuracy_fn=accuracy_fn,
                         )

print(model_res)

# save(model)
trained_model = FashionMNISTModelV2(input_shape=1,
                            output_shape=10,
                            hidden_units=128,).to(device)

trained_model.load_state_dict(torch.load(f=MODEL_SAVING_PATH))

# model_res = eval_model(model=trained_model,
#                          data_loader=test_dataloader,
#                          loss_fn=loss_fn,
#                          accuracy_fn=accuracy_fn,
#                          )

# print(model_res)

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)
            

test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)



def plot_data(sample, labels, right_labels):
    rows, cols = 3, 3
    fig = plt.figure(figsize=(9, 9))

    for i in range(1, rows*cols+1):

        fig.add_subplot(rows, cols, i)
        plt.imshow(sample[i-1].squeeze(), cmap='gray')
        pred_l = class_names[labels[i-1]]
        true_l = class_names[right_labels[i-1]]
        if pred_l == true_l:
            plt.title(f'Pred: {pred_l} Truth: {true_l}', c='g')
        else: 
            plt.title(f'Pred: {pred_l} Truth: {true_l}', c ='r')
        plt.axis('off')
        plt.xlabel(False)

    plt.show()

# plot_data(test_saples, test_labels)
preb_probs = make_predictions(model=trained_model,
                              data=test_samples,
                              device=device)

pred_classes = preb_probs.argmax(dim=1)

# print(pred_classes)

plot_data(test_samples, pred_classes, test_labels)
