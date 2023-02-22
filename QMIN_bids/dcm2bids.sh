#!/bin/bash

module load miniconda
source activate dcm2bids 

# scripts directory
scriptsDir=passian-processing/QMIN_bids 
# read the relevant inputs
inputDir=/directory/where/BIDS/are
# setup directories in BIDS format
outputDir=/directory/for/BIDS/outputs/


for sub in `cat ${scriptsDir}/IDs.txt` ; do
  cd ${inputDir}/${sub} && arr=(*) && cd -
  h1="${arr[0]:0:8}" # to ensure the session data is picked up
  echo "Processing $sub ses-$h1"
  dcm2bids -d ${inputDir}/${sub}/2* -p${sub} -s ses-$h1 -c ${scriptsDir}/dcm2bids_config_final.json -o ${outputDir} --clobber
  echo "$sub complete"
done
