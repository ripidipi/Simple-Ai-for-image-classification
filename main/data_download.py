import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pathlib
import matplotlib.pyplot as plt
from shutil import make_archive
from pathlib import Path

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

train_data_stl = datasets.STL10(root=data_dir,
                                split="train",
                                download=state,
                                target_transform=None)

test_data_stl = datasets.STL10(root=data_dir,
                               split = "test",
                                download=state,
                                target_transform=None)


class_names_food = train_data_food.classes
class_names_stl = train_data_stl.classes


def plot_data(ind:int, cur_set:datasets) -> None:
    class_names = cur_set.classes
    img, label = cur_set[ind]
    plt.imshow(img, cmap='gray')
    plt.title(class_names[label])
    plt.axis('off')
    plt.xlabel(False)

    plt.show()



# plot_data0()
# plot_data(5000, train_data_food)
# print(class_names_food)
# print(class_names_stl)


