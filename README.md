# Deep-Feature-Selection
The repository of paper "Deep Feature Selection for Anomaly Detection Based on Pretrained Network and Gaussian Discriminative Analysis"
# 运行环境
1. 系统：Ubuntu 16.04.5 LTS
2. CUDA版本：10.0, V10.0.130
3. 显卡：GeForce RTX 2080 Ti
4. 包管理工具：conda
5. 软件包：详见gaussian.yaml文件
6. 特别说明：‘Bearing_data_generator’和‘show_cam_images’两个文件夹中的文件为MATLAB文件，运行环境为MATLAB 2018b

# 数据集
1. MVTec AD：来自https://www.mvtec.com/company/research/datasets/mvtec-ad/
2. Bearing Data来自：https://engineering.case.edu/bearingdatacenter/welcome

# 运行示例
## 复现类激活图
在ubuntu系统的终端激活所需环境后，进入cam所在文件夹，输入运行get_img_cam文件
```
conda acntivate gaussian
cd <your cam folder path>
python -m get_img_cam
```
## 复现文献中的AUROC数据：
### 单类测试
进入gaussian文件夹，打开ubuntu终端，输入：
  ```
  conda acntivate gaussian
  cd gaussian
  python -m src.common.trainer --model gaussian --category bottle --arch efficientnet-b4 --extract_blocks 0 1 2 3 4 5 6 7 8 --max_nb_epochs 0 --gpus 0 --npca --variance_threshold 0.99 --select_space 2_3
  ```
### 数据集测试
进入gaussian文件夹，打开ubuntu终端，输入：
  ```
  conda activate gaussian
  cd <your gaussian folder path>
  python -m src.scripts.my_table1 --gpu --logpath <your log path>
  ```
