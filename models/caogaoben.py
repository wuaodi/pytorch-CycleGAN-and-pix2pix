# # obtain the edge of an image by opencv
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
#
# def edge_loss(image1,image2):
#     image1 = np.array(image1.cpu()) #this method will lose the grad of pytorch tensor
#     image2 = np.array(image2.cpu())
#     image1_blur = cv2.GaussianBlur(image1[0,0,:,:], (5, 5), 1)
#     image2_blur = cv2.GaussianBlur(image2[0,0,:,:], (5, 5), 1)
#     image1_canny = cv2.Canny(image1_blur, 30, 150)
#     image2_canny = cv2.Canny(image2_blur, 30, 150)
#     loss = np.mean((image1_canny-image2_canny)**2) #mse
#     return loss


import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('../imgs/opt.jpg').convert('L')
im = np.array(im, dtype='float32')
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(im.astype('uint8'), cmap='gray')

im1 = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
print(im1.shape)
conv1 = nn.Conv2d(1, 1, 3, bias=False)
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
conv1.weight.data = torch.from_numpy(sobel_kernel)
edge1 = conv1(Variable(im1))
print(edge1.shape)
edge1 = edge1.data.squeeze().numpy()
print(edge1.shape)
plt.subplot(1, 2, 2)
plt.imshow(edge1, cmap='gray')
plt.show()