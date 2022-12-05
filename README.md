# Deep Feature Selection

This repository contains information and code from the paper "[Deep Feature Selection for Anomaly Detection Based on Pretrained Network and Gaussian Discriminative Analysis](https://ieeexplore.ieee.org/document/9887794)" published at IEEE Open Journal of Instrumentation and Measurement, vol. 1, Art. no. 3500611, by Jie Lin, Song Chen, Enping Lin, and Yu Yang. 

Please note that the codes in "gaussian' are a modified version of the codes from: https://github.com/ORippler/gaussian-ad-mvtec (reference paper:  ["Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection"](https://arxiv.org/abs/2005.14140) presented at ICPR2020 and its journal extension ["Gaussian Anomaly Detection by Modeling the Distribution of Normal Data in Pretrained Deep Features"](https://ieeexplore.ieee.org/abstract/document/9493210) published at IEEE TIM)

## Development Environment

- System: Ubuntu 16.04.5 LTS

- CUDA version：10.0, V10.0.130

- GPU: GeForce RTX 2080 Ti

- Package: please check gaussian.yaml for more details

- The files in the two folders -- ‘Bearing_data_generator’ and ‘show_cam_images’ are developed in MATLAB 2018b

## Datasets

1. MVTec AD: downloaded from https://www.mvtec.com/company/research/datasets/mvtec-ad/
2. CWRU Bearing Data: downloaded from https://engineering.case.edu/bearingdatacenter/welcome

## Examples

### Train on a certain category

```
conda activate <your environment name>
cd <your gaussian folder path>
python -m src.common.trainer --model gaussian --category bottle --arch efficientnet-b4 --extract_blocks 0 1 2 3 4 5 6 7 8 --max_nb_epochs 0 --gpus 0 --npca --variance_threshold 0.99 --select_space 2_3
```

### Test on the dataset

```
conda activate <your environment name>
cd <your gaussian folder path>
python -m src.scripts.my_table1 --gpu --logpath <your log path>
```

### CAM Visualization

```
conda activate <your environment name>
cd <your cam folder path>
python -m get_img_cam
```

Copy the results of 'get_img_cam' and paste it to the folder 'show_cam_images/cam_images', and then run the MATLAB code 'show_cam_images.m' to get the image.

### Preprocessing of the CWRU data

Download the bearing data (in .mat form) to the 'Bearing_data_generator' folder, and run the MATLAB code 'preproc_data_lj.m'.

## License

Copyright (C) 2022 by Xiamen University (https://www.xmu.edu.cn/)

License: AGPL (GNU Affero General Public License) open source license