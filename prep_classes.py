''' This reads subject class labels from a .csv file and rearranges them to classes by directories'''
'''The image files are expected to be in a CAPS (BIDS) format, preproc'd with Aramis Clinica t1-linear'''
import pandas as pd
import os, sys
from glob import glob
from helper_functions import load_adni


root_dir = '/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_ADNI_bl'  # the full adni baseline dataset
print(root_dir)
labels_csv = "/home/imber/Projects/PASSIAN/data/ADNI/aramis_preproc/CAPS_smallsample/ADNIMERGE_2022-09-02.csv"


image_files_list, image_class, cn = load_adni(root_dir, labels_csv)
