# copy from https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py

import argparse
import cv2
import numpy as np
import torch
import scipy.io as io
import os
from torchvision import models
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image


# # 如果出现 OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--block', type=int, default=0,
                        help='level to view')
    parser.add_argument('--mat_path', type=str, default=None,
                        help='npz file path')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    # 在这里设置模型
    model = EfficientNet.from_pretrained('efficientnet-b4')

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    block = args.block
    target_layer = model._blocks[block]

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.use_cuda)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    # cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)

    # Here grayscale_cam has only one image in the batch
    # grayscale_cam = grayscale_cam[0, :]

    # 保存为mat文件
    io.savemat(os.path.join(args.mat_path,'block{}.mat'.format(args.block)), {'grayscale_cam': grayscale_cam})