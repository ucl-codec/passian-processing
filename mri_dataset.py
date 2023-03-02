''' Dataset class for ADNI data'''
import torch, os
import pandas as pd
from glob import glob

# Note, use only baseline images here
class MRIDataset_bl(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

    def __len__(self):
        return len(self.image_files)