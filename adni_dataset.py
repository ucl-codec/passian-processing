''' Dataset class for ADNI data'''
import torch, os
import pandas as pd
from glob import glob

# Note, use only baseline images here
class ADNIDataset_bl(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels_csv, transforms):
        self.image_dir = image_dir
        # Notes: This is the CAPS preprocd dir from aramis clinidaDL; can later change ses-bl to include more t-pts
        self.image_files = sorted(glob(os.path.join(self.image_dir, 'subjects/*/ses-bl/t1_linear/*Crop*.nii.gz')))
        self.labels_csv = labels_csv
        labelss = pd.read_csv(self.labels_csv, low_memory=False)[['PTID', 'DX', 'VISCODE']]  # only ids, dx, visit
        labelss['FID'] = 'sub-' + labelss['PTID'].str.replace('_', '')  # add a new file ID columns
        labelss = labelss[labelss['VISCODE'] == 'bl']
        self.labels = labelss[labelss['FID'].isin([file.split('/')[10] for file in self.image_files])]
        self.transforms = transforms

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

    def __len__(self):
        return len(self.image_files)