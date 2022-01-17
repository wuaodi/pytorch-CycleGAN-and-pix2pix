# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
import torch
import math
import torch.nn as nn

### pytorch实现laplacian变换
# def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
#     # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
#     x_coord = torch.arange(kernel_size)
#     x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
#     y_grid = x_grid.t()
#     xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
#
#     mean = (kernel_size - 1) / 2.
#     variance = sigma ** 2.
#
#     # Calculate the 2-dimensional gaussian kernel which is
#     # the product of two gaussian distributions for two different
#     # variables (in this case called x and y)
#     gaussian_kernel = (1. / (2. * math.pi * variance)) * \
#                       torch.exp(
#                           -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
#                           (2 * variance)
#                       )
#
#     # Make sure sum of values in gaussian kernel equals 1.
#     gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
#
#     # Reshape to 2d depthwise convolutional weight
#     gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
#     gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
#
#     gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
#                                 bias=False, padding=kernel_size // 2)
#
#     gaussian_filter.weight.data = gaussian_kernel
#     gaussian_filter.weight.requires_grad = False
#
#     return gaussian_filter
#
# g1 = get_gaussian_kernel(kernel_size=5, sigma=2, channels=1)
# print(g1)
#
#
# img = torch.randn([1,3,64,64]).cuda()
# blur_layer = get_gaussian_kernel().cuda()
#
# blured_img = blur_layer(img)
# print(blured_img.shape)







# ###使用pytorch实现laplacian卷积
# import numpy as np
# import torch
# from torch import nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# from PIL import Image
# import matplotlib.pyplot as plt
#
# im = Image.open('../imgs/opt.jpg').convert('L')
# im = np.array(im, dtype='float32')
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(im.astype('uint8'), cmap='gray')
#
# im1 = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
# print(im1.shape) # torch.Size([1, 1, 720, 1280])
#
# conv1 = nn.Conv2d(1, 1, 3, bias=False)
# sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
# sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
# conv1.weight.data = torch.from_numpy(sobel_kernel)
# edge1 = conv1(Variable(im1))
# print(edge1.shape) # torch.Size([1, 1, 720, 1280])
#
# edge1 = edge1.data.squeeze().numpy()
# print(edge1.shape) # (718, 1278)
#
# plt.subplot(1, 2, 2)
# plt.imshow(edge1, cmap='gray')
# plt.show()








### 开始定义的边缘损失，直接用numpy和cv2的函数是不行的，没有pytorch的梯度
# # obtain the edge of an image by opencv
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn

# def edge_loss(image1,image2):
#     image1 = np.array(image1.cpu()) #this method will lose the grad of pytorch tensor
#     image2 = np.array(image2.cpu())
#     image1_blur = cv2.GaussianBlur(image1[0,0,:,:], (5, 5), 1)
#     image2_blur = cv2.GaussianBlur(image2[0,0,:,:], (5, 5), 1)
#     image1_canny = cv2.Canny(image1_blur, 30, 150)
#     image2_canny = cv2.Canny(image2_blur, 30, 150)
#     loss = np.mean((image1_canny-image2_canny)**2) #mse
#     return loss
