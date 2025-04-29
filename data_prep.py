# Script to move 15% of XML files from train/ to devel/

import os
import random
import shutil
import math

# Parameters
train_dir = 'data/Train'
devel_dir = 'data/Devel'

# Create devel directory if it doesn't exist
os.makedirs(devel_dir, exist_ok=True)

# List all XML files in train/
files = [f for f in os.listdir(train_dir) if f.endswith('.xml')]

# Calculate 15% of the files
n_files_to_move = math.ceil(len(files) * 0.15)

# Randomly select files to move
files_to_move = random.sample(files, n_files_to_move)

# Move selected files
for file_name in files_to_move:
    src_path = os.path.join(train_dir, file_name)
    dest_path = os.path.join(devel_dir, file_name)
    shutil.move(src_path, dest_path)

print(f"Moved {n_files_to_move} files from {train_dir} to {devel_dir}")