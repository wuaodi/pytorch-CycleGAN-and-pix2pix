import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def edge_loss(image1,image2):
    im1 = torch.mean(image1, 1, keepdim=True, out=None) # color image 2 gray image
    im2 = torch.mean(image2, 1, keepdim=True, out=None)
    conv1 = nn.Conv2d(1, 1, 3, bias=False)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') #Laplacian
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    conv1.weight.data = torch.from_numpy(sobel_kernel).to(device)
    edge1 = conv1(Variable(im1))
    edge2 = conv1(Variable(im2))
    loss = torch.mean((edge1-edge2)**2) #mse
    return loss