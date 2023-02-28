''' Dataset class for QMINMC data'''
import torch, os
import pandas as pd
from glob import glob

# Note, use only baseline images here
class QMINDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

        ## The below is another methods if the labels csv is passed to the dataloader. Currently handled in main script
        # image = self.transforms(self.image_files[index])
        # print('Image is ', self.image_files[index].split('/')[10])
        # label = self.labels['DX'][self.labels['FID'] == self.image_files[index].split('/')[10]].item()  # this should return image FID
        # print('Label is', label)
        # return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)
