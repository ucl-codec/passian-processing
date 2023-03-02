'''This is from the tutorial
https://github.com/Project-MONAI/tutorials/blob/main/3d_classification/densenet_training_array.ipynb'''
import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from mri_dataset import MRIDataset_bl
from helper_functions import load_adni
from monai.networks.nets import DenseNet121


pin_memory = torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
root_dir = '/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_ADNI_bl'  # the full adni baseline dataset
labels_csv = "/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_smallsample/ADNIMERGE_2022-09-02.csv"

image_files_list, image_class, cn = load_adni(root_dir, labels_csv)
val_transforms = Compose(
    [LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

num_class = 3
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
best_model_name = "22_02_2023_16:57_best_metric_model.pth"

model = DenseNet121(spatial_dims=3, in_channels=1,
                    out_channels=num_class).to(device)
model.load_state_dict(torch.load(
    os.path.join(root_dir, best_model_name)))
model.eval()

## Prep train, val, test
val_frac = 0.2
test_frac = 0.1
length = len(image_files_list)
num_class = 3
indices = np.arange(length)
np.random.shuffle(indices)
bs = 5  # batch size
max_epochs = 2  # AUC 0.7741 after 100 epochs

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

# create a validation data loader
test_ds = MRIDataset_bl(image_files=image_files_list[-10:], labels=image_class[-10:], transforms=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)  # , pin_memory=torch.cuda.is_available()
itera = iter(test_loader)

def get_next_im():
    test_data = next(itera)
    return test_data[0].to(device), test_data[1].unsqueeze(0).to(device)

def plot_occlusion_heatmap(im, heatmap):
    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(im.cpu()))
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.show()

# Get a random image and its corresponding label
img, label = get_next_im()


# Get the occlusion sensitivity map
occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, mask_size=10, n_batch=10, stride=12)
# Only get a single slice to save time.
# For the other dimensions (channel, width, height), use
# -1 to use 0 and img.shape[x]-1 for min and max, respectively
depth_slice = img.shape[2] // 2
occ_sens_b_box = [depth_slice - 1, depth_slice, -1, -1, -1, -1]

occ_result, _ = occ_sens(x=img, b_box=occ_sens_b_box)
occ_result = occ_result[0, label.argmax().item()][None]

fig, axes = plt.subplots(1, 2, figsize=(25, 15), facecolor="white")

for i, im in enumerate([img[:, :, depth_slice, ...], occ_result]):
    cmap = "gray" if i == 0 else "jet"
    ax = axes[i]
    im_show = ax.imshow(np.squeeze(im[0][0].detach().cpu()), cmap=cmap)
    ax.axis("off")
    fig.colorbar(im_show, ax=ax)

plt.show()