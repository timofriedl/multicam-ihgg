import torch
import torch.nn as nn
from torch import Tensor


class CoordConv(nn.Module):
    cc_cache = {}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 **kwargs):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.size()

        # Coordinate channels
        channel_x, channel_y = CoordConv.create_coordinate_channels(batch_size, width, height)
        x = torch.cat([x, channel_x.type_as(x), channel_y.type_as(x)], dim=1)
        x = self.conv(x)

        return x

    @staticmethod
    def create_x_y_grids(width, height) -> (Tensor, Tensor):
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        x_grid, y_grid = torch.meshgrid(x, y)
        return x_grid.transpose(0, 1), y_grid.transpose(0, 1)

    @staticmethod
    def create_coordinate_channels(batch_size, width, height) -> (Tensor, Tensor):
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
        CoordConv.cc_cache = {}


class CoordBDVAE(nn.Module):
    def __init__(self, width, height, latent_dim, channels=3):
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

    def encode(self, x):
        return torch.chunk(self.enc_net(x), 2, dim=1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        batch_size = z.size(0)

        z = z.view(z.shape + (1, 1))
        z = z.expand(-1, -1, self.height, self.width)

        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

        x = self.dec_net(x)

        return x

    def forward(self, x: Tensor):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)

        return x_rec, mu, logvar
