import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import math


def get_gaussian_layer(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depth wise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel.cuda()
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def get_laplacian_layer():
    laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  # Laplacian
    laplacian_kernel = laplacian_kernel.reshape((1, 1, 3, 3))
    laplacian_filter = nn.Conv2d(1, 1, 3, bias=False)
    laplacian_filter.weight.data = torch.from_numpy(laplacian_kernel).cuda()
    laplacian_filter.weight.requires_grad = False
    return laplacian_filter


def edge_loss(image1, image2):
    im1 = torch.mean(image1, 1, keepdim=True, out=None)  # color image 2 gray image
    im2 = torch.mean(image2, 1, keepdim=True, out=None)

    blur_layer = get_gaussian_layer(kernel_size=5, sigma=2, channels=1)
    laplacian_layer = get_laplacian_layer()

    blured_im1 = blur_layer(Variable(im1))
    blured_im2 = blur_layer(Variable(im2))

    edge_im1 = laplacian_layer(blured_im1)
    edge_im2 = laplacian_layer(blured_im2)

    loss = torch.mean((edge_im1 - edge_im2) ** 2)  # mse
    return loss
