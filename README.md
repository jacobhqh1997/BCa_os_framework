# ***An integrated deep learning system for prognostication prediction in bladder cancer***

© This code is made available for non-commercial academic purposes. 

## Overview
Precise stratification of survival risk for bladder cancer (BCa) is critical for accurate personalized therapy. Here we present an interpretable and  integrated deep learning system for prognostication prediction in bladder cancer. We trained and validated two prognostic networks, MacroVisionNet and UniVisionNet. Inspired by the previous research "PathFinder", six potential BCa prognostic biomarkers were explored and evaluated.

## Directory Structure


* **Training Scripts**: *Training Scripts for BlaPaSeg, MacroVisionNet and UniVisionNet.*
* **Data_process**: *Data preprocessing file.*
* **Feature_extractor**: *macro, micro feature extraction.*
* **Network**: *MacroVisionNet and UniVisionNet  structure.*


## Pre-requisites and Environment

### Our Environment
* Linux (Tested on Ubuntu 22.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 4090 x 2)
* Python (3.9.12), PyTorch (version 2.0.0), Lifelines (version 0.27.8), NumPy (version 1.24.1), Pandas (version 2.1.2), Albumentations (version 1.3.1), OpenCV (version 4.8.1), Pillow (version 9.3.0), OpenSlide (version 1.1.2), Captum (version 0.6.0), SciPy (version 1.11.3), Seaborn (version 0.13.0), Matplotlib (version 3.8.1), torch_geometric (version 2.4.0), torch-scatter (version 2.1.2), torch-sparse (version 0.6.18).
### Environment Configuration
1. Create a virtual environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).
   ```
   $ conda create -n env python=3.9.12
   $ conda activate env
   $ pip install torch
   ```
      *Note:  `pip install` command is required for Pytorch installation.*
   
2. To try out the Python code and set up environment, please activate the `env` environment first:

   ``` shell
   $ conda activate env
   ```
3. For ease of use, you can just set up the environment and run the following:
   ``` shell
   $ pip install -r requirements.txt
   ```

## Data Preparation

### Data Format
* WSIs and clinical information of patients are used in this project. Raw WSIs are stored as ```.svs```, ```.mrxs``` or ```.tiff``` files. Clinical information are stored as ```.csv``` files. 

### Generate Macro npy file

* WSIs are first processed by BlaPaSeg network to get multi-class tissue probability heatmaps , which sorted as ```.npy``` files.
  The macro mode of WSIs can be by generated by calling:

    ``` shell
    $ cd ./Data_process
    $ python BlaPaSeg_inference.py
    ```
* To cut the empty area of macro mode and get square input for training, call:
    ``` shell
    $ cd ./Data_process
    $ python cut_heatmap.py
    ```

### Generate feature pt file

* Tumor patch feature files are extracted based on Macro npy file and WSIs
  The  macro prognostic  feature of WSIs can be by running macro_feature.py 

    ``` shell
    $ cd ./Feature_extractor
    $ python inference_ctan.py   #get CTransPath feature
    $ python inference_giga.py   #get Prov-Gigapath feature
    $ python inference_uni.py    #get Uni feature
    $ python inference_virchow.py  #get Virchow feature
    $ python inference_virchow.py  #get Virchow feature
    $ python macro_feature.py   # get macro prognostic features
    
    ```
  

### Generate final Micro pt file

* The final Micro pt file is generated by combining tumor patch feature files and macro prognostic features file

  ```bash
  $ cd ./Data_process
  $ python feature_combine.py
  ```

### Training Scripts

```shell
$ cd ./Training Scripts
$ python BlaPaSeg_cash.py   # BlaPaSeg training scripts 
$ python train_macro_cash.py   # MacroVisionNet training scripts 
$ python train_uni_cash.py   # UniVisionNet training scripts 
```

### Data Distribution

```bash
DATA_ROOT/
    └──DATASET/
         ├── clinical_information                       + + + 
                ├── train.csv                               +
                ├── valid.csv                               +
                └── ...                                     +
         ├── WSI_data                                       +
                ├── train                                   +
                       ├── slide_1.svs                      +
                       ├── slide_2.svs                Source WSI file
                       └── ...                              +
                ├──valid                                    +
                       ├── slide_1.svs                      +
                       ├── slide_2.svs                      +
                       └── ...                              +
                └── ...                                 + + +
         ├── macro_file                                 + + +
                ├── train                                   +
                       ├── slide_1.npy                      +
                       ├── slide_2.npy                      +
                       └── ...                              +
                ├── valid                                   +
                       ├── slide_1.npy                      +
                       ├── slide_2.npy                      +
                       └── ...                              +
                └── ...                                     +
         └── maicro_file                                    +
                ├── train                                   +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +
                ├── valid                                   +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +            
```
DATA_ROOT is the base directory of all datasets (e.g. the directory to your SSD or HDD). DATASET is the name of the folder containing data specific to one experiment.


## Acknowledgements
- Prognosis training and test code base structure was inspired by [[PathFinder](https://github.com/Biooptics2021/PathFinder)](https://github.com/mahmoodlab/PathomicFusion).

- Integrated Muscle Tumor Score (IMTS) was inspired by [TILAb-Score](https://github.com/TissueImageAnalytics/TILAb-Score).

  



