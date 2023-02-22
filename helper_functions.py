import pandas as pd
import os, sys, torch
from glob import glob


def load_adni(root_dir, csv):
    # Notes: root_dir should be CAPS_preprocessed; csv is the main csv with the ADNI dataset
    image_files_list = sorted(glob(os.path.join(root_dir, 'subjects/*/ses-bl/t1_linear/*Crop*.nii.gz')))
    subjects = [file.split('/')[10] for file in image_files_list]

    # Filtering and sorting the labels from the clinical data csv
    labels_df = pd.read_csv(csv, low_memory=False)[['PTID', 'DX', 'VISCODE']]  # only ids, dx, visit
    labels_df['FID'] = 'sub-' + labels_df['PTID'].str.replace('_', '')  # add a new file ID columns
    labels_df = labels_df[labels_df['VISCODE'] == 'bl']
    labels_df = labels_df[labels_df['FID'].isin(subjects)]  # filter
    labels_df = labels_df.dropna(subset=['DX'])  # get rid of NaNs in the labels
    labels_df = labels_df.sort_values('FID')  # this should work - test on larger sample!
    print('Class balance on entire dataset (train, validation, test):\n', labels_df['DX'].value_counts())
    no_labels = [x for x in subjects if
                 x not in labels_df['FID'].values]  # these subjects are missing labels (delete from dataset)
    if no_labels:
        print('The following subjects in the image_files_list are missing labels:\n', no_labels)
        print('Aborting...')
        sys.exit()

    cn = ['CN', 'MCI', 'Dementia']  # class names
    cd = {cn[0]: 0.0, cn[1]: 1.0, cn[2]: 2.0}  # class dictionary  # note to self: always start at 0
    labels_df = labels_df.replace({'DX': cd})
    image_class = torch.tensor(list(labels_df['DX']))
    image_class = image_class.type(torch.LongTensor)

    return image_files_list, image_class, cn

# Todo @marcella: Write a load_qminmc function to do the above dataset-specific operations for QMIN-MC
