# Deep-Feature-Selection
The repository of paper "Deep Feature Selection for Anomaly Detection Based on Pretrained Network and Gaussian Discriminative Analysis"
# 运行环境
1. 系统：Ubuntu 16.04.5 LTS
2. CUDA版本：10.0, V10.0.130
3. 显卡：GeForce RTX 2080 Ti
4. 包管理工具：conda
5. 软件包：详见gaussian.yaml文件

# 数据集
1. MVTec AD：来自https://www.mvtec.com/company/research/datasets/mvtec-ad/
2. Bearing Data来自：https://engineering.case.edu/bearingdatacenter/welcome

# 运行示例
1. 复现类激活图：在ubuntu系统的终端激活所需环境后，进入cam所在文件夹，输入运行get_img_cam文件
```
conda acntivate gaussian
cd cam
python -m get_img_cam
```
