import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from vision_model import model, Food
from dataset_setting import train_dataloader_custom, test_dataloader_custom, class_names
from save_and_load import save
from train_and_test import train_step, test_step
from save_and_load import MODEL_SAVING_PATH

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using : {device}")


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


NUM_EPOCHS = 3

# trained_model = Food(input_shape=3,
#                hidden_units=32,
#                output_shape=len(class_names)).to(device)

# trained_model.load_state_dict(torch.load(MODEL_SAVING_PATH))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

from timeit import default_timer as timer 

start_time = timer()
model_result = train(model=model,
                    train_dataloader=test_dataloader_custom,
                    test_dataloader=test_dataloader_custom,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=NUM_EPOCHS,
                    )
end_time = timer()

save(model)

print(f"Total training time: {end_time-start_time:.3f} seconds")
