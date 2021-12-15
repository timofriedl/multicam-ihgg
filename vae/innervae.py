import torch
import torch.nn as nn
from torch import Tensor


class InnerVae(nn.Module):
    """
    An implementation of beta-VAE for the second stage compression of EncodeConcatEncodeVae
    """

    def __init__(self, input_dim: int, latent_dim: int):
        """
        Creates a new InnerVae instance

        :param input_dim: the number of dimensions of the data to compress
        :param latent_dim: the number of latent dimensions in the encoded vector
        """
        super(InnerVae, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Real output size is twice the number of latent dimensions
        output_dim = latent_dim * 2

        # Encoder
        self.enc_net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=(input_dim + output_dim) // 2),
            nn.ReLU(),
            nn.Linear(in_features=(input_dim + output_dim) // 2, out_features=(input_dim + 3 * output_dim) // 4),
            nn.ReLU(),
            nn.Linear(in_features=(input_dim + 3 * output_dim) // 4, out_features=output_dim)
        )

        # Decoder
        self.dec_net = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=(input_dim + 3 * latent_dim) // 4),
            nn.ReLU(),
            nn.Linear(in_features=(input_dim + 3 * latent_dim) // 4, out_features=(input_dim + latent_dim) // 2),
            nn.ReLU(),
            nn.Linear(in_features=(input_dim + latent_dim) // 2, out_features=input_dim)
        )

    def encode(self, x: Tensor) -> (Tensor, Tensor):
        """
        Compresses a given batch of input vectors to its encoded distributions

        :param x: a pytorch tensor with shape [batch_size, input_dim] to compress
        :return: a pytorch tensor of mu values, and a pytorch tensor of logvar values that represent the latent vectors
        """
        return torch.chunk(self.enc_net(x), 2, dim=1)

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Samples a batch of latent vectors from its representing mu and logvar values

        :param mu: a pytorch tensor of mu values from the encode function
        :param logvar: a pytorch tensor of logvar values from the encode function
        :return: a pytorch tensor of latent vectors with shape [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """
        Decodes a given batch of latent vectors back to a reconstruction of the input vectors

        :param z: the pytorch tensor of latent vectors with shape [batch_size, latent_dim]
        :return: the pytorch tensor of reconstruction vectors with shape [batch_size, input_dim]
        """
        return self.dec_net(z)

    def forward(self, x: Tensor) -> Tensor:
        """
        Encodes a given batch of input vectors to their latent representations
        and decodes them back to a reconstruction of the original vectors.

        :param x: the pytorch tensor of input vectors with shape [batch_size, input_dim]
        :return: the pytorch tensor of reconstruction vectors with same shape
        """
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)

        return x_rec, mu, logvar
