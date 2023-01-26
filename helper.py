import os
import shutil


def merge_cats_dogs(cat_dir: str, dog_dir: str):
    os.makedirs("dataset", exist_ok=True)
    for i in os.listdir(cat_dir):
        shutil.copy(os.path.join(cat_dir, i), os.path.join("dataset", "cats_" + i))
    for i in os.listdir(dog_dir):
        shutil.copy(os.path.join(dog_dir, i), os.path.join("dataset", "dogs_" + i))
