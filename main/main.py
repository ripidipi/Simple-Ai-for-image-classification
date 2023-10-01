import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from vision_model import Food
from train_and_test import train_step, test_step
from dataset_setting import class_names, quick_test_dataloader
from save_and_load import MODEL_SAVING_PATH

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using : {device}")

trained_model = Food(input_shape=3,
               hidden_units=16,
               output_shape=len(class_names)).to(device)

trained_model.load_state_dict(torch.load(MODEL_SAVING_PATH))

loss_fn = nn.CrossEntropyLoss()

test_loss, test_acc = test_step(model=trained_model, 
                                dataloader=quick_test_dataloader,
                                loss_fn=loss_fn)

print(test_loss, test_acc)


