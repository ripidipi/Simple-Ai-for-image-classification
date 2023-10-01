import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pathlib
import matplotlib.pyplot as plt
from shutil import make_archive
from pathlib import Path
import random

data_dir = pathlib.Path(r"C:\Users\ripid\Desktop\python\Ai\pytorch_tests\data")
data_path = Path("data/")
image_path = data_path / "food"

if image_path.is_dir():
    # print(f"{image_path} directory, skipping download")
    state = False
    pass
else: 
    print(f"{image_path} is a new directory of data, downloading")
    state = True
    image_path.mkdir(parents=True, exist_ok=True)

train_data_food = datasets.Food101(root=data_dir,
                            split="train",
                            download=state,
                            target_transform=None)

test_data_food = datasets.Food101(root=data_dir,
                           split="test",
                           download=state,
                           target_transform=None)

def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          n: int = 10,
                          display_shape: bool = True):
    
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        plt.subplot(2, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        # if dataset.classes:
        #     title = f"class: {dataset.classes[targ_label]}"
        #     if display_shape:
        #         title = title + f"\nshape: {targ_image_adjust.shape}"
        # plt.title(title)
        plt.xlabel(False)

    plt.show()

# display_random_images(train_data_place)