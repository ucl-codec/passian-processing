import os
import random
import shutil
from tqdm import tqdm

# Define the paths to the source and destination folders
source_folder = "/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_ADNI_bl"
destination_folder_1 = "/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/A_node_caps_adni_bl"
destination_folder_2 = "/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/B_node_caps_adni_bl"

# Create the destination folders if they don't already exist
if not os.path.exists(destination_folder_1):
    os.makedirs(destination_folder_1)
if not os.path.exists(destination_folder_2):
    os.makedirs(destination_folder_2)

# Set the number of subjects to sample and the random seed
num_samples = 300
random_seed = 42

# Create a list of all the files in the source folder and its subfolders
all_files = []
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if "Crop" in file and file.endswith(".nii.gz"):
            all_files.append(os.path.join(root, file))

# Shuffle the list of files
random.seed(random_seed)
random.shuffle(all_files)

# Take the first num_samples files and copy them to the two destination folders,
# preserving the folder structure
for i, file_path in tqdm(enumerate(all_files[:num_samples*2])):
    relative_path = os.path.relpath(file_path, source_folder)
    if i < num_samples:
        destination_path = os.path.join(destination_folder_1, relative_path)
    else:
        destination_path = os.path.join(destination_folder_2, relative_path)
    destination_directory = os.path.dirname(destination_path)
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    shutil.copy2(file_path, destination_path)