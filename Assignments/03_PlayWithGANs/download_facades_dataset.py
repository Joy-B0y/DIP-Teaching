import os
import urllib.request
import tarfile
from pathlib import Path

FILE = "facades"
URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{FILE}.tar.gz"
TAR_FILE = f"./datasets/{FILE}.tar.gz"
TARGET_DIR = f"./datasets/{FILE}/"

os.makedirs(TARGET_DIR, exist_ok=True)
print(f"Downloading {URL} dataset to {TARGET_DIR}...")

urllib.request.urlretrieve(URL, TAR_FILE)
print(f"Dataset downloaded to {TAR_FILE}")


print("Extracting the dataset...")
with tarfile.open(TAR_FILE, "r:gz") as tar:
    tar.extractall(path="./datasets/")
print(f"Dataset extracted to {TARGET_DIR}")

os.remove(TAR_FILE)
print(f"Removed tar file: {TAR_FILE}")

train_list_path = Path("./train_list.txt")
val_list_path = Path("./val_list.txt")

train_images = sorted(Path(f"{TARGET_DIR}/train").glob("*.jpg"))
val_images = sorted(Path(f"{TARGET_DIR}/val").glob("*.jpg"))

print("Generating train and validation file lists...")
with train_list_path.open("w") as train_file:
    train_file.writelines([str(img) + "\n" for img in train_images])

with val_list_path.open("w") as val_file:
    val_file.writelines([str(img) + "\n" for img in val_images])

print(f"Train file list saved to {train_list_path}")
print(f"Validation file list saved to {val_list_path}")
