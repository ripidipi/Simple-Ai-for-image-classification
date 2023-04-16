#saving our model 
from pathlib import Path
import torch


# create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
# create model save path
MODEL_NAME = "01_pytorch_model_0.pth"
MODEL_SAVING_PATH = (f"{MODEL_PATH}/{MODEL_NAME}")

def save(model):    
    print(f"Saving path for model is {MODEL_SAVING_PATH}")
    # save the models state dict
    torch.save(obj=model.state_dict(),   
                f=MODEL_SAVING_PATH)

    