'''Data exploration for the clinical data'''
import pandas as pd
import os
from glob import glob

dates = pd.read_csv('../data/CODEC/clin/dates_all.txt', delim_whitespace=True)
scan_vars = pd.read_csv('../data/CODEC/clin/ScanVariablesCODEC_ORIG.csv')
scan_vars_comp = pd.read_csv('../data/CODEC/clin/ScanVariablesCODEC_Comp.csv')
clin = pd.read_csv('../data/CODEC/clin/NeurocogDatabaseLONG_ANON.csv')

actual_cases = os.listdir('../data/CODEC/BIDS')
actual_cases_formatted = [case.strip('sub-') for case in actual_cases]

conversion = pd.read_csv('../data/CODEC/clin/CODEC_scan_ID_key.csv').set_index('research ID')
conv_dict = conversion['patient ID'].to_dict()
clin.insert(1, 'mapped_PATID', clin["PAH_ID"].replace(conv_dict))
