import torch
import torch.nn as nn
from torch import Tensor

"""
An implementation of CoordBDVAE that combines a Spatial Broadcast Decoder VAE with Coordinate Convolutional Layers.

Original author: Aleksandar Aleksandrov
Heavily modified by Timo Friedl
"""


class CoordConv(nn.Module):
    """
    A Coordinate Convolutional Layer that uses two orthogonal gradient channels to provide positional information.
    """

    # A cache for coordinate channels
    cc_cache = {}

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0,
                 **kwargs):
        """
        Creates a new CoordConv layer.

        :param in_channels: the number of input channels, e.g. 3 for a RGB image
        :param out_channels: the number of output channels
        :param kernel_size: the CNN kernel size
        :param stride: the CNN stride
        :param padding: the CNN padding
        """
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forwards a given batch of images through the CoordConv layer.

        :param x: the batch of images to forward with shape [batch_size, 3, height, width]
        :return: the output of this CoordConv layer
        """
        batch_size, _, height, width = x.size()

        # Coordinate channels
        channel_x, channel_y = CoordConv.create_coordinate_channels(batch_size, width, height)
        x = torch.cat([x, channel_x.type_as(x), channel_y.type_as(x)], dim=1)
        x = self.conv(x)

        return x

    @staticmethod
    def create_x_y_grids(width: int, height: int) -> (Tensor, Tensor):
        """
        Creates two matrices,
        one with an horizontal gradient from 0.0 to 1.0, and
        one with a vertical gradient from 0.0 to 1.0

        :param width: the horizontal size of the output matrices
        :param height: the vertical size of the output matrices
        :return: the horizontal gradient matrix, and the vertical gradient matrix
        """
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        x_grid, y_grid = torch.meshgrid(x, y)
        return x_grid.transpose(0, 1), y_grid.transpose(0, 1)

    @staticmethod
    def create_coordinate_channels(batch_size: int, width: int, height: int) -> (Tensor, Tensor):
        """
        Creates the coordinate channels for one batch of input images
        using cached channels if possible.

        :param batch_size: the number of input images
        :param width: the horizontal size of the input images
        :param height: the vertical size of the input images
        :return: the 3D tensor of horizontal gradients, and the 3D tensor of vertical gradients
        """
        cache_key = (batch_size, width, height)
        if cache_key in CoordConv.cc_cache:
            return CoordConv.cc_cache[cache_key]

        x_grid, y_grid = CoordConv.create_x_y_grids(width, height)
        channel_x = x_grid.repeat(batch_size, 1, 1, 1)
        channel_y = y_grid.repeat(batch_size, 1, 1, 1)

        CoordConv.cc_cache[cache_key] = (channel_x, channel_y)
        return channel_x, channel_y

    @staticmethod
    def empty_cache():
        """
        Deletes the gradient matrix cache
        """
        CoordConv.cc_cache = {}


class CoordBDVAE(nn.Module):
    """
    A Spatial Broadcast Decoder VAE with Coordinate Convolutional Layers
    """

    def __init__(self, width: int, height: int, latent_dim: int, channels=3):
        """
        Creates a new CoordBDVAE.

        :param width: the horizontal size of the input images
        :param height: the vertical size of the input images
        :param latent_dim: the number of latent dimensions
        :param channels: the number of image channels
        """
        super(CoordBDVAE, self).__init__()

        self.width = width
        self.height = height
        self.latent_dim = latent_dim

        # Encoder
        self.enc_net = nn.Sequential(
            CoordConv(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=64 * (width // 16) * (height // 16), out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=latent_dim * 2)
        )

        # Decoder
        self.dec_net = nn.Sequential(
            CoordConv(in_channels=latent_dim + 2, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            CoordConv(in_channels=64, out_channels=channels, kernel_size=3, padding=1)
        )

        x_grid, y_grid = CoordConv.create_x_y_grids(width, height)
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

    def encode(self, x: Tensor) -> Tensor:
        """
        Compresses a given batch of images to its encoded distributions

        :param x: a pytorch tensor of images with shape [batch_size, 3, height, width] to compress
        :return: a pytorch tensor of mu values, and a pytorch tensor of logvar values that represent the latent vectors
        """
        return torch.chunk(self.enc_net(x), 2, dim=1)

    def sample(self, mu, logvar) -> Tensor:
        """
        Samples a batch of latent vectors from its representing mu and logvar values

        :param mu: a pytorch tensor of mu values from the encode function
        :param logvar: a pytorch tensor of logvar values from the encode function
        :return: a pytorch tensor of latent vectors with shape [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z) -> Tensor:
        """
        Decodes a given batch of latent vectors back to a reconstruction of the input images

        :param z: the pytorch tensor of latent vectors with shape [batch_size, latent_dim]
        :return: the pytorch tensor of reconstruction images with shape [batch_size, 3, height, width]
        """
        batch_size = z.size(0)

        z = z.view(z.shape + (1, 1))
        z = z.expand(-1, -1, self.height, self.width)

        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

        x = self.dec_net(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Encodes a given batch of input images to their latent representations
        and decodes them back to a reconstruction of the original images.

        :param x: the pytorch tensor of input images with shape [batch_size, 3, height, width]
        :return: the pytorch tensor of reconstruction images with same shape
        """
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)

        return x_rec, mu, logvar
