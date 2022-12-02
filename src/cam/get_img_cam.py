
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '4'  # 设置显卡

# 数据集中的categorise：类别
available_categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

# efficient中2-8层MBConvBlock与stage的关系，即
# block_stage_b0 = np.array([0, 1, 2, 2, 3, 3, 4, 1])
# block_stage_b1 = np.array([0, 2, 3, 3, 4, 4, 5, 2])
# block_stage_b2 = np.array([0, 2, 3, 3, 5, 5, 6, 2])
# block_stage_b3 = np.array([0, 2, 4, 4, 6, 6, 8, 2])
block_stage_b4 = np.array([0, 2, 4, 4, 6, 6, 8, 2])
# block_stage_b5 = np.array([0, 3, 5, 5, 7, 7, 9, 3])
# block_stage_b6 = np.array([0, 3, 6, 6, 8, 8, 11, 3])
# block_stage_b7 = np.array([0, 4, 7, 7, 10, 10, 13, 4])
# 选择需要的模型
block_stage = block_stage_b4.cumsum()

# 选择cam的方法
method = 'gradcam'

# 出于文件大小考虑这里只保留了了bottle类下broken_large中000.png对应的cam图
# 如果需要计算其他类category_root_path这个文件夹下的所有编号为000.png的图片，那么注释掉“if dir != "broken_large": continue”这句话即可
# 如果需要计算所有的图片，那么注释掉target_imgs_num = 1，把target_imgs_num = len(imgs)还原即可
for category in available_categories:
    print(category + ":开始")
    # 数据集中图片的category路径
    category_root_path = '/DATA_Inter/lj/backup/dataset/original/images/PublicDataset/Mvtec_AD/' + category + '/test/'
    # 存放cam图片的跟路径
    cam_root_path = '/home/lj/Work/cam_images'
    # cam图片的category路径
    cam_category_path = os.path.join(cam_root_path,category)
    os.makedirs(cam_category_path,exist_ok=True)
    for root,dirs,files in os.walk(category_root_path):
        for dir in dirs:
            # 选择dir，注释掉这一段就是计算所有类的所有dir
            if dir != "broken_large": # 只用某个文件夹里的图片做演示
                continue
            # 数据集中图片的dir路径
            original_category_dir_path = os.path.join(root,dir)
            # cam图片的dir路径
            cam_category_dir_path = os.path.join(cam_category_path,dir)
            os.makedirs(cam_category_dir_path,exist_ok=True)
            for imgs_root,[],imgs in os.walk(original_category_dir_path):
                imgs.sort()
                # 选择需要计算cam图的数量
                # target_imgs_num = len(imgs)
                target_imgs_num = 5
                for img in range(target_imgs_num):
                    print(category + "-->" + dir + "-->" + imgs[img])
                    # 数据集中具体图片的路径
                    original_category_dir_img_path = os.path.join(original_category_dir_path,imgs[img])
                    # cam图关于具体数据集图片的路径
                    cam_category_dir_img_path = os.path.join(cam_category_dir_path,imgs[img])
                    os.makedirs(cam_category_dir_img_path,exist_ok=True)
                    # cam图关于具体数据集图片的levels路径
                    cam_category_dir_img_block_path = os.path.join(cam_category_dir_img_path,'blocks')
                    os.makedirs(cam_category_dir_img_block_path, exist_ok=True)
                    for i in range(block_stage[-1]):
                        print('python cam.py --method {} --image_path {} --mat_path {} --block {}'.format(method, original_category_dir_img_path, cam_category_dir_img_block_path, i))
                        os.system('python cam.py --method {} --image_path {} --mat_path {} --block {}'.format(method, original_category_dir_img_path, cam_category_dir_img_block_path, i))
    print(category + ":完成")
