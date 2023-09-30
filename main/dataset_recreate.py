import os
from pathlib import Path
import random
import pathlib
import shutil

data_dir = Path("data/")
data_path = data_dir / "food" / "images"
target_classes = []

for cur in os.listdir(data_path):
    target_classes.append(cur)

def get_subset(image_path=data_path, 
               data_splits=["train", "test"],
               target_classes=target_classes,
               amount = 0.9):
    lable_splits = {}

    for data_split in data_splits:
        lable_path = data_dir / "food" / "meta" / f"{data_split}.txt"
        with open(lable_path, "r") as f:
            lables = [line.strip('\n') for line in f.readlines() 
                      if line.split('/')[0] in target_classes]

        number_to_sample = round(len(lables) * amount)
        samples = random.sample(lables, k=number_to_sample)
        image_paths = [Path(str(image_path / sample) + ".jpg") for sample in samples]
        lable_splits[data_split] = image_paths
    return lable_splits

lable_splits = get_subset()
# print(lable_splits["train"][:10])
        
target_dir_name = data_dir / "food60"

target_dir = Path(target_dir_name)

target_dir.mkdir(parents=True, exist_ok=True)

for image_split in lable_splits.keys():
    for image_path in lable_splits[str(image_split)]:
        dest_dir = target_dir / image_split / image_path.parent.stem / image_path.name
        if not dest_dir.parent.is_dir():
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Copying {image_path} to {dest_dir}...")
        try:
            shutil.copy2(image_path, dest_dir)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"{e}")