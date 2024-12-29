import pandas as pd
import os, sys, torch
from glob import glob
import nibabel as nib

def make_bl_labels_adni(root_dir, csv_file):
    print("-----Preparing labels csv for baseline ADNI-----")
    # Notes: root_dir should be CAPS_preprocessed; csv is the main csv with the ADNI dataset
    image_files_list = sorted(glob(os.path.join(root_dir, 'subjects/*/ses-bl/t1_linear/*Crop*.nii.gz')))
    subjects = [file.split('/')[-4] for file in image_files_list]

    # Filtering and sorting the labels from the clinical data csv
    labels_df = pd.read_csv(csv_file, low_memory=False)[['PTID', 'DX', 'VISCODE']]  # only ids, dx, visit
    labels_df['FID'] = 'sub-' + labels_df['PTID'].str.replace('_', '')  # add a new file ID columns
    labels_df = labels_df[labels_df['VISCODE'] == 'bl']
    labels_df = labels_df[labels_df['FID'].isin(subjects)]  # filter
    labels_df = labels_df.dropna(subset=['DX'])  # get rid of NaNs in the labels
    labels_df = labels_df.sort_values('FID')  # this should work - test on larger sample!
    print('Class balance on entire dataset (train + validation + test): \n', labels_df['DX'].value_counts())
    no_labels = [x for x in subjects if
                 x not in labels_df['FID'].values]  # these subjects are missing labels (delete from dataset)
    if no_labels:
        print('The following subjects in the image_files_list are missing labels:\n', no_labels)
        print('You should delete them from the dataset of images...')
    else:
        print('All subject image files seem to have labels')

    cleanlabelsave = os.path.join(root_dir, 'labels.csv')
    print('Saving labels file to ', cleanlabelsave)
    cleanlabels_df = labels_df[['FID', 'DX']]
    cleanlabels_df.to_csv(cleanlabelsave, index=False)


def load_labels(root_dir):
    image_files_list = sorted(glob(os.path.join(root_dir, 'subjects/*/ses-bl/t1_linear/*Crop*.nii.gz')))
    labels_df = pd.read_csv(os.path.join(root_dir, 'labels.csv'))
    cn = ['CN', 'MCI', 'Dementia']  # class names
    cd = {cn[0]: 0.0, cn[1]: 1.0, cn[2]: 2.0}  # class dictionary  # note to self: always start at 0
    labels_df = labels_df.replace({'DX': cd})
    image_class = torch.tensor(list(labels_df['DX']))
    image_class = image_class.type(torch.LongTensor)
    return image_files_list, image_class, cn

def make_bl_labels_qmin(root_dir, csv_file):
    print("-----Preparing labels csv for baseline QMIN-MC-----")
    # Notes: root_dir should be CAPS_preprocessed; csv is the main csv with the QMIN dataset
    image_files_list = sorted(glob(os.path.join(root_dir, 'subjects/sub*/ses*/t1_linear/*Crop*.nii.gz')))
    subjects = [file.split('/')[-4] for file in image_files_list]

    # Filtering and sorting the labels from the clinical data csv
    labels_df = pd.read_csv(csv_file, low_memory=False)[['FID','DX','sex','age_bl']]   # adjusted the names as in ADNI's file
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
    diagnosis_dict = {'DX': new_class_names}
    labels_df = labels_df.replace(diagnosis_dict)

    # Remove rows with diagnosis values currently not in use
    valid_diagnoses = ['CN', 'MCI', 'Dementia']
    mask = labels_df['DX'].isin(valid_diagnoses)
    labels_df = labels_df[mask]
    labels_df = labels_df.reset_index(drop=True)   # reset the index for the new df

    print('Class balance on entire dataset (train, validation, test):\n', labels_df['DX'].value_counts())
    no_labels = [x for x in subjects if
                 x not in labels_df['FID'].values]  # these subjects are missing labels (delete from dataset)
    if no_labels:
        print('The following subjects in the image_files_list are missing labels:\n', no_labels)
        print('You should delete them from the dataset of images...')
    else:
        print('All subject image files seem to have labels')

    cleanlabelsave = os.path.join(root_dir, 'labels.csv')
    print('Saving labels file to ', cleanlabelsave)
    cleanlabels_df = labels_df[['FID', 'DX']]
    cleanlabels_df.to_csv(cleanlabelsave, index=False)

def get_images_from_dirs(data_dir):
    # This is in the case we are using PyTorch class folder <-> each class in its folder
    # basically from here: https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb
    class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
    num_classes = len(class_names)
    image_files = [
        [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
        for i in range(num_classes)
    ]
    num_each = [len(image_files[i]) for i in range(num_classes)]
    image_files_list = []
    image_classes = []
    for i in range(num_classes):
        image_files_list.extend(image_files[i])
        image_classes.extend([i] * num_each[i])
    num_total = len(image_classes)
    # check the dims of the first nifti (assuming they are all the same)
    image = nib.load(image_files_list[0])
    data = image.get_fdata()
    shape = data.shape
    spacing = image.header.get_zooms()

    print(f"Total image count: {num_total}")
    print("Based on the first image: ")
    print(f"Image dimensions: {shape}'")
    print(f'Image spacing: {spacing}')
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")

    return image_files_list, image_classes, class_names, num_classes

## Just testing, to be deleted
roodir = '/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/A_node_caps_adni_bl'
csv = '/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/A_node_caps_adni_bl/ADNIMERGE_2022-09-02.csv'

make_bl_labels_adni(roodir, csv)
image_files_list, image_class, cn = load_labels(roodir)

# roodir = '/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/A_node_caps_adni_bl'
# csv = '/home/imber/Projects/PASSIAN/data/qmin_participants.csv'
# make_bl_labels_qmin(roodir, csv)