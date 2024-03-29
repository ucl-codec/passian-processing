# passian-processing
Neuroimaging processing pipelines for PASSIAN, a [CODEC](https://ucl-codec.github.io) project

Note: It's possible to consider using a lighter touch for image preprocessing, e.g., using [MONAI](https://docs.monai.io/) or [TorchIO](https://torchio.readthedocs.io/) for data loading and/or augmentation.

## BIDS Apps for PASSIAN

- For Preparing T1w (& T2w) MRI, e.g., for deep learning
  - QC: 
    - [mriqc](https://github.com/nipreps/mriqc) 
    - [BrainQCNet](https://github.com/garciaml/BrainQCNet) ([paper](https://www.biorxiv.org/content/10.1101/2022.03.11.483983v1.full); GPU version: [BrainQCNet_GPU](https://github.com/garciaml/BrainQCNet_GPU))
  - [smriprep](https://github.com/ucl-codec/smriprep)
  - [TorchIO](https://torchio.readthedocs.io/transforms/preprocessing.html)
  - [MRtrix3](https://mrtrix.readthedocs.io/en/latest/) - skull stripping, etc.


## Other BIDS Apps of potential interest

- MRI prep, e.g., for deep learning
  - [SPM](https://github.com/bids-apps/SPM)
- MRI processing, e.g., for [Disease Progression Modelling](https://ucl-pond.github.io)
  - [freesurfer](https://github.com/e-dads/freesurfer)
  - [ANTS cortical thickness](https://github.com/bids-apps/antsCorticalThickness)
  - [mindboggle](https://github.com/bids-apps/mindboggle)
  - [multiscalebrainparcellator](https://github.com/sebastientourbier/multiscalebrainparcellator)

### Other considerations
- [MONAI](https://monai.io/), e.g., for [FL](https://github.com/Project-MONAI/tutorials/tree/main/federated_learning/openfl) using OpenFL
- [ClinicaDL](https://github.com/aramis-lab/clinicaDL)
- [TVB Pipeline](https://github.com/McIntosh-Lab/tvb-ukbb)
