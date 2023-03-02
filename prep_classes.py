''' This reads subject class labels from a .csv file and rearranges them to classes by directories'''
'''The image files are expected to be in a CAPS (BIDS) format, preproc'd with Aramis Clinica t1-linear'''
import pandas as pd
import os, sys, shutil
from glob import glob
from helper_functions import load_adni
from tqdm import tqdm

root_dir = '/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_ADNI_bl'  # the full adni baseline dataset
print(root_dir)
labels_csv = "/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_smallsample/ADNIMERGE_2022-09-02.csv"
rearranged_dir = '/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_ADNI_bl_classdirs'


image_files_list, image_class, cn = load_adni(root_dir, labels_csv)

for class_name in cn:
    sub_folder_path = os.path.join(rearranged_dir, class_name)
    os.makedirs(sub_folder_path, exist_ok=True)

# Copy files to subdirectory based on class name
for (file_name, label) in tqdm(zip(image_files_list, image_class)):
    # src_file_path = os.path.join(file_folder_path, file_name)
    if int(label) == 0:
        dest_file_path = os.path.join(rearranged_dir, cn[0])
        shutil.copy(file_name, dest_file_path)
    elif int(label) == 1:
        dest_file_path = os.path.join(rearranged_dir, cn[1])
        shutil.copy(file_name, dest_file_path)
    elif int(label) == 2:
        dest_file_path = os.path.join(rearranged_dir, cn[2])
        shutil.copy(file_name, dest_file_path)
    else:
        print(f"No label found for {file_name.split('/')[-1]}, skipping...")
