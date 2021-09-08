import torch
import torch.nn as nn


class InnerVae(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(InnerVae, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

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

    def encode(self, x):
        return torch.chunk(self.enc_net(x), 2, dim=1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec_net(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_rec = self.decode(z)

        return x_rec, mu, logvar
