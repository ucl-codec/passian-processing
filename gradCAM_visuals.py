import os, sys
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
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
from monai.visualize.class_activation_maps import GradCAM

root_dir = '/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_ADNI_bl'  # the full adni baseline dataset
print(root_dir)
num_class = 3
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
best_model_name = "22_02_2023_16:57_best_metric_model.pth"

model = DenseNet121(spatial_dims=3, in_channels=1,
                    out_channels=num_class).to(device)

## Evaluate model on test dataset
model.load_state_dict(torch.load(
    os.path.join(root_dir, best_model_name)))
model.eval()

cam = GradCAM(nn_module=model, target_layers="class_layers.relu")
oneimage = "/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_ADNI_bl/subjects/sub-002S0816/ses-bl/t1_linear/sub-002S0816_ses-bl_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"

result = cam(x=torch.rand((1, 1, 169, 208, 179)).to(device))

# y_true = []
# y_pred = []


# with torch.no_grad():
#     for test_data in test_loader:
#         test_images, test_labels = (
#             test_data[0].to(device),
#             test_data[1].to(device),
#         )
#         assert isinstance(model(test_images).argmax, object)
#         pred = model(test_images).argmax(dim=1)
#         for i in range(len(pred)):
#             y_true.append(test_labels[i].item())
#             y_pred.append(pred[i].item())
#
# ## Print classification report
# print(classification_report(
#     y_true, y_pred, target_names=cn, digits=4))
