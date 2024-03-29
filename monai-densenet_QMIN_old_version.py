''' This is from this Fed-BioMed tutorial, but locally, without FL
https://gitlab.inria.fr/fedbiomed/fedbiomed/-/blob/develop/notebooks/monai-2d-image-classification-gpu.ipynb
https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb'''


import os
import shutil
import tempfile
#import matplotlib.pyplot as plt
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

# print_config()

## Setup data dir
directory = '/home/mm2075/rds/hpc-work/PASSIAN/QMINCaps'  # os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

data_dir = directory

## Set deterministic
set_determinism(seed=42)


# Notes: This is the CAPS preprocd dir from aramis clinidaDL; can later change ses-bl to include more t-pts
image_files_list = sorted(glob(os.path.join(data_dir, 'subjects/sub*/ses*/t1_linear/*Crop*.nii.gz')))

## Prep train, val, test
val_frac = 0.2
test_frac = 0.1
length = len(image_files_list)
num_class = 3
indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

labels_csv = "/home/mm2075/rds/hpc-work/PASSIAN/QMINDatabase/ClinicalData/qmin_participants.csv"
labels_df = pd.read_csv(labels_csv, low_memory=False)[['participant_id','sex','age_bl', 'diagnosis']]  # only ids, dx, visit
#labels_df['FID'] = 'sub-' + labels_df['PTID'].str.replace('_', '')  # add a new file ID columns
#labels_df = labels_df[labels_df['VISCODE'] == 'bl']
#labels_df = labels_df[labels_df['FID'].isin([file.split('/')[10] for file in image_files_list])]  # filter
#labels_df = labels_df.sort_values('FID')  # this should work - test on larger sample!
print('Class balance on entire dataset (train, validation, test):\n', labels_df['diagnosis'].value_counts())

# Define the new class names and the corresponding dictionary
new_class_names = {'Vascular dementia': 'Dementia',
                   'Mixed dementia (Alzheimer\'s disease and vascular dementia)': 'Dementia',
                   'Alzheimer\'s disease': 'Dementia',
                   'Frontotemporal dementia - behavioural variant': 'Dementia',
                   'Unspecified dementia': 'Dementia',
                   'Frontotemporal dementia - non-fluent variant aphasia': 'Dementia',
                   'Alzheimer\'s disease - Posterior Cortical Atrophy': 'Dementia',
                   'Dementia with Lewy Bodies': 'Dementia',
                   'Alzheimer\'s disease - Logopenic aphasia': 'Dementia',
                   'Frontotemporal dementia - semantic dementia': 'Dementia',
                   'F03X Unspecified dementia' : 'Dementia',
                   'Mild Cognitive Impairment': 'MCI',
                   'Functional Memory Disorder': 'CN',
		   'No diagnosis (normal)' : 'CN',
		   'Healthy relative' : 'CN',
		   'Alcohol related dementia' : 'Degenerative',
		   'Circumscribed brain atrophy' : 'Degenerative',
	  'Benign familial tremor and unexpected visuospatial dysfunction' : 'Degenerative',
		   'basal ganglia disorder' : 'Degenerative',
		   'G31.01 Non-fluent variant primary progressive aphasia (nfvPPA) with some subclinical corticobasal features' : 'Degenerative',
		   'C9orf72 carrier becoming early symptomatic' : 'Degenerative',
		   'Parkinson\'s disease' : 'Degenerative',
		   'Corticobasal Syndrome' : 'Degenerative',
		   'Meningioma' : 'Degenerative',
		   'Unknown' : 'Non-degenerative',
		   'Depression' : 'Non-degenerative',
		   'Stroke' : 'Non-degenerative',
		   'Anxiety' : 'Non-degenerative',
		   'Psychosis' : 'Non-degenerative',
		   'Adjustment Disorder' : 'Non-degenerative',
                   'Epilepsy with memory problems' : 'Non-degenerative',
		   'Learning difficulties' : 'Non-degenerative',
		   'Traumatic brain injury' : 'Non-degenerative',
		   'Uncertain' : 'Non-degenerative'}

# Replace the old labels with new ones
diagnosis_dict = {'diagnosis': new_class_names}
labels_df = labels_df.replace(diagnosis_dict)
print(labels_df['diagnosis'].value_counts())

# Remove rows with diagnosis values currently not in use
valid_diagnoses = ['CN', 'MCI', 'Dementia']
mask = labels_df['diagnosis'].isin(valid_diagnoses)
labels_df = labels_df[mask]
# reset the index for the new df
labels_df = labels_df.reset_index(drop=True)

# Convert the new labels to numerical values
cn = ['CN','MCI', 'Dementia']  # class names 
cd = {cn[0]: 0.0, cn[1]: 1.0, cn[2]: 2.0}  # class dictionary
labels_df = labels_df.replace({'diagnosis': cd})
print('Final labels used:\n',
labels_df['diagnosis'].value_counts())

# Convert the labels to a PyTorch tensor
image_class = torch.tensor(list(labels_df['diagnosis']))
image_class = image_class.type(torch.LongTensor)

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

## Define train transforms (augmentations)
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ]
)

val_transforms = Compose(
    [LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])

bs = 2  # batch size

## Define dataset and dataloader 
train_ds = MRIDataset_bl(train_x, train_y, train_transforms)
train_loader = DataLoader(
    train_ds, batch_size=bs, shuffle=True, num_workers=10)

val_ds = MRIDataset_bl(val_x, val_y, val_transforms)
val_loader = DataLoader(
    val_ds, batch_size=bs, num_workers=10)

test_ds = MRIDataset_bl(test_x, test_y, val_transforms)
test_loader = DataLoader(
    test_ds, batch_size=bs, num_workers=10)



## Define network and optimiser
# note, random transforms each epoch means different data each epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=3, in_channels=1,
                    out_channels=num_class).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
max_epochs = 2  # AUC 0.7741 after 100 epochs
val_interval = 1
auc_metric = ROCAUCMetric()

## Train model
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        # print(batch_data[0].size(), ' and label', batch_data[1].size())
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
        	f"{step}/{len(train_ds) // train_loader.batch_size}, "
        	f"train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                # now = datetime.now().strftime("%d_%m_%Y_%H:%M") # removed as this did not work on the HPC
                torch.save(model.state_dict(), os.path.join(
                    root_dir, f"best_metric_model.pth")) # removed the date_time part
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")



## Plot model and metric
#plt.figure("train", (12, 6))
#plt.subplot(1, 2, 1)
#plt.title("Epoch Average Loss")
#x = [i + 1 for i in range(len(epoch_loss_values))]
#y = epoch_loss_values
#plt.xlabel("epoch")
#plt.plot(x, y)
#plt.subplot(1, 2, 2)
#plt.title("Val AUC")
#x = [val_interval * (i + 1) for i in range(len(metric_values))]
#y = metric_values
#plt.xlabel("epoch")
#plt.plot(x, y)
#plt.show()

## Evaluate model on test dataset
model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        assert isinstance(model(test_images).argmax, object)
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())

## Print classification report
print(classification_report(
    y_true, y_pred, target_names=cn, digits=4))
