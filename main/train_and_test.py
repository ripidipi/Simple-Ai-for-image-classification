import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from vision_model import model
from dataset_setting import train_dataloader_custom, test_dataloader_custom
from save_and_load import save


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using : {device}")



def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):
    train_loss, train_acc = 0, 0

    model.train()

    for batch, (img_train, label) in enumerate(dataloader):
        
        img_train, label = img_train.to(device), label.to(device)

        model.eval()

        # forward pass
        y_pred = model(img_train)

        # loss calculation
        loss = loss_fn(y_pred, label)
        train_loss += loss

        # train_acc += accuracy_fn(y_true=label,
        #                          y_pred=y_pred.argmax(dim=1))
    
        # zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optimizer
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==label).sum().item()/len(y_pred)

    train_loss /= len(train_dataloader_custom)
    train_acc /= len(train_dataloader_custom)
    print(f' Train loss: {train_loss:.4f} \nTrain accuracy: {train_acc:.2f}')
    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    model.eval() 

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

