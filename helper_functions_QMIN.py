import pandas as pd
import os, sys, torch
from glob import glob


def load_qmin(root_dir, csv):
    # Notes: root_dir should be CAPS_preprocessed; csv is the main csv with the QMIN dataset
    image_files_list = sorted(glob(os.path.join(root_dir, 'subjects/sub*/ses*/t1_linear/*Crop*.nii.gz')))
    subjects = [file.split('/')[8] for file in image_files_list] # for QMIN

    # Filtering and sorting the labels from the clinical data csv
    labels_csv = "/home/mm2075/rds/hpc-work/PASSIAN/QMINDatabase/ClinicalData/qmin_participants.csv"
    labels_df = pd.read_csv(labels_csv, low_memory=False)[['participant_id','sex','age_bl', 'diagnosis']]   # add a new file ID columns
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

    print('Class balance on entire dataset (train, validation, test):\n', labels_df['diagnosis'].value_counts())
    no_labels = [x for x in subjects if
                 x not in labels_df['participant_id'].values]  # these subjects are missing labels (delete from dataset)
    if no_labels:
        print('The following subjects in the image_files_list are missing labels:\n', no_labels)
        print('Aborting...')
        sys.exit()

    # Convert the new labels to numerical values
    cn = ['CN','MCI', 'Dementia']  # class names 
    cd = {cn[0]: 0.0, cn[1]: 1.0, cn[2]: 2.0}  # class dictionary
    labels_df = labels_df.replace({'diagnosis': cd})
    print('Final labels used:\n',
    labels_df['diagnosis'].value_counts())

    # Convert the labels to a PyTorch tensor
    image_class = torch.tensor(list(labels_df['diagnosis']))
    image_class = image_class.type(torch.LongTensor)

    return image_files_list, image_class, cn
