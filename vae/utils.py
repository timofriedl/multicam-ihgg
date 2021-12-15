import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.utils import make_grid, save_image

"""
A bunch of helpful functions for VAE training and general tensor / array conversions.

Original author: Aleksandar Aleksandrov
Heavily modified by Timo Friedl
"""

# The device used for VAE training
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def images_to_tensor(images: np.ndarray) -> Tensor:
    """
    Converts a numpy array of rgb images with shape [num_images, height, width, 3]
    to a normalized pytorch tensor with shape [num_images, 3, height, width]
    using the given device from above

    :param images: the numpy array of images to convert
    :return: the converted pytorch tensor
    """
    images = images.transpose((0, 3, 1, 2)) / 255.0
    images = torch.tensor(images, dtype=torch.float32)
    images = images.to(device)
    return images


def images_to_tensor_cpu(images: np.ndarray) -> Tensor:
    """
    Converts a numpy array of rgb images with shape [num_images, height, width, 3]
    to a normalized pytorch tensor with shape [num_images, 3, height, width]
    using cpu as device

    :param images: the numpy array of images to convert
    :return: the converted pytorch tensor
    """
    images = images.transpose((0, 3, 1, 2)) / 255.0
    images = torch.tensor(images, dtype=torch.float32)
    images = images.to('cpu')
    return images


def np_to_tensor(array: np.ndarray) -> Tensor:
    """
    Converts a numpy array to a corresponding float32 pytorch tensor
    using the given device from above

    :param array: the numpy array to convert
    :return: the converted pytorch tensor
    """
    return torch.tensor(array, dtype=torch.float32).to(device)


def tensor_to_np(tensor: Tensor) -> np.ndarray:
    """
    Converts a pytorch tensor to a corresponding numpy array

    :param tensor: the pytorch tensor to convert
    :return: the converted numpy array
    """
    return tensor.detach().cpu().numpy()


def image_to_tensor(image: np.ndarray) -> Tensor:
    """
    Converts a single rgb image with shape [height, width, 3]
    to a pytorch tensor with shape [1, 3, height, width]
    that represents a one-element batch of images
    using the device from above

    :param image: the numpy array to convert
    :return: the converted pytorch tensor
    """
    images = image.copy()
    images = images.reshape((1, image.shape[0], image.shape[1], 3))
    images = images.transpose((0, 3, 1, 2)) / 255.0
    images = torch.tensor(images, dtype=torch.float32)
    images = images.to(device)
    return images


def tensor_to_image(tensor: Tensor) -> np.ndarray:
    """
    Converts a pytorch tensor with shape [1, 3, height, width]
    that represents a one-element batch of images
    to a numpy array with shape [height, width, 3] representing the same image

    :param tensor: the pytorch tensor to convert
    :return: the converted numpy array
    """
    image = tensor.detach().cpu().numpy()
    image[(image < 0.)] = 0.
    image[(image > 1.)] = 1.
    image = np.floor(255 * image).astype(np.uint8).reshape(tensor.shape[1:])
    return image.transpose(1, 2, 0)


def save_examples(examples: Tensor, name: str, img_format="jpeg"):
    """
    Saves a batch of training images to a given location

    :param examples: the batch of images with shape [num_images, 3, height, width]
    :param name: the base name of the output file
    :param img_format: the image format as a string, use "png" for lossless compression
    """
    clipped = torch.clamp(examples.detach(), 0, 1)
    image = make_grid(clipped)
    save_image(image, "images/{0}.{1}".format(name, img_format))


def loss_fn(x: Tensor, x_rec: Tensor, mu: Tensor, logvar: Tensor, alpha=10., beta=1.) -> (float, float, float):
    """
    The VAE training loss function.

    :param x: the batch of original input images
    :param x_rec: the batch of reconstructions of the input images
    :param mu: the batch of mu values from the network encoding
    :param logvar: the batch of logvar values from the network encoding
    :param alpha: a weight factor hyperparameter
    :param beta: a weight factor hyperparameter
    :return: the weighted mean square error, kl loss, and sum of both
    """
    batch_size = x.size(0)

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    mse_loss = F.mse_loss(x_rec, x, reduction='none')
    mse_loss = mse_loss.view(batch_size, -1)
    mse_loss = mse_loss.sum(dim=-1)
    mse_loss = mse_loss.mean()

    return alpha * mse_loss, beta * kl, alpha * mse_loss + beta * kl


def loss_fn_weighted(x: Tensor, x_rec: Tensor, mu: Tensor, logvar: Tensor, alpha=10., beta=1.) -> (float, float, float):
    """
    Another VAE training loss function with special weight factor.

    :param x: the batch of original input images
    :param x_rec: the batch of reconstructions of the input images
    :param mu: the batch of mu values from the network encoding
    :param logvar: the batch of logvar values from the network encoding
    :param alpha: a weight factor hyperparameter
    :param beta: a weight factor hyperparameter
    :return: the weighted mean square error, kl loss, and sum of both
    """
    batch_size = x.size(0)

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    mse_loss = F.mse_loss(x_rec, x, reduction='none')
    a = torch.ones_like(mse_loss)
    a[:, :, 40:, :] = 2.
    mse_loss = mse_loss * a
    mse_loss = mse_loss.view(batch_size, -1)
    mse_loss = mse_loss.sum(dim=-1)
    mse_loss = mse_loss.mean()

    return alpha * mse_loss, beta * kl, alpha * mse_loss + beta * kl


def loss_fn_weighted2(x: Tensor, x_rec: Tensor, mu: Tensor, logvar: Tensor, alpha=10., beta=1.) -> (
        float, float, float):
    """
    Another VAE training loss function with different special weight factor.

    :param x: the batch of original input images
    :param x_rec: the batch of reconstructions of the input images
    :param mu: the batch of mu values from the network encoding
    :param logvar: the batch of logvar values from the network encoding
    :param alpha: a weight factor hyperparameter
    :param beta: a weight factor hyperparameter
    :return: the weighted mean square error, kl loss, and sum of both
    """
    batch_size = x.size(0)

    if len(x.shape) == 2:
        x = x.view(batch_size, 1, 1, -1)
        x_rec = x_rec.view(batch_size, 1, 1, -1)

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    mse_loss = F.mse_loss(x_rec, x, reduction='none')
    a = torch.ones_like(mse_loss)
    a[:, :, :60, :] = 2.
    mse_loss = mse_loss * a
    mse_loss = mse_loss.view(batch_size, -1)
    mse_loss = mse_loss.sum(dim=-1)
    mse_loss = mse_loss.mean()

    return alpha * mse_loss, beta * kl, alpha * mse_loss + beta * kl


def tensor_wrap(x: Tensor) -> Tensor:
    """
    Wraps a given pytorch tensor inside another brackets to increase the dimensionality by one.

    Example:
    tensor_wrap(tensor([1., 2., 3.])) = tensor([[1., 2., 3.]])

    :param x: the pytorch tensor to wrap
    :return: the wrapped pytorch tensor
    """
    return x.unsqueeze(0)
