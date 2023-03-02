''' Runner script to (locally) run training '''
import os, sys
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121, EfficientNetBN
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
from mri_dataset import MRIDataset_bl
from glob import glob
import pandas as pd
from collections import Counter
from datetime import datetime
from helper_functions import load_adni
from helper_functions import get_images_from_dirs

root_dir = '/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_ADNI_bl_classdirs'  # the full adni baseline dataset
print("Dataset directory is: ", root_dir)
## Set deterministic
set_determinism(seed=317)

# image_files_list = sorted(glob(os.path.join(root_dir, '*/*Crop*.nii.gz')))
image_files_list, image_class, class_names, num_classes = get_images_from_dirs(root_dir)
subjects = [file.split('/')[-1][:12] for file in image_files_list]  # subject ID is the first 13 characters of the file

# Todo: abstract these for tidyness

## Prep train, val, test
val_frac = 0.2
test_frac = 0.1
length = len(image_files_list)
num_class = 3
indices = np.arange(length)
np.random.shuffle(indices)
bs = 10  # batch size
max_epochs = 300  # AUC 0.7741 after 100 epochs

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]
test_x = [image_files_list[i] for i in test_indices]
test_y = [image_class[i] for i in test_indices]

print(
    f"Training count: {len(train_x)}, Validation count: "
    f"{len(val_x)}, Test count: {len(test_x)}")
print('Class balance on training population:', torch.tensor(train_y).unique(return_counts=True))
print('Class balance on validation population:', torch.tensor(val_y).unique(return_counts=True))
print('Class balance on test population:', torch.tensor(test_y).unique(return_counts=True))



## Todo: implement 5-fold crossval


## Todo: clean up imports