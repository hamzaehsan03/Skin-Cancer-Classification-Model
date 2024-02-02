import os
import shutil
import numpy as np

folders = ['1', '0']
for folder in folders: 
    os.makedirs(f'data/validation/{folder}', exist_ok=True)

def count_files(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])

def split_data_validation(source, destination, split_size, seed=727):
    files = os.listdir(source)
    np.random.seed(seed)
    np.random.shuffle(files)
    split_point = int(split_size * len(files))
    move_files = files[:split_point]

    for file in move_files:
        shutil.move(os.path.join(source, file), os.path.join(destination, file))
    
split_ratio = 0.2
seed = 727

for folder in folders:
    print(f"Training {folder}: {count_files(f'data/train/{folder}')}")
    print(f"Validation {folder}: {count_files(f'data/validation/{folder}')}")


for folder in folders:
    source_dir = f'data/train/{folder}'
    destination_dir = f'data/validation/{folder}'
    split_data_validation(source_dir, destination_dir, split_ratio, seed)

for folder in folders:
    print(f"Training {folder}: {count_files(f'data/train/{folder}')}")
    print(f"Validation {folder}: {count_files(f'data/validation/{folder}')}")
