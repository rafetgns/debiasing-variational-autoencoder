from dataclasses import dataclass

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))


def build_encoder(in_channels, n_outputs, image_size=64, n_filters=12):
    feat = image_size // 16
    return nn.Sequential(
        ConvBlock(in_channels, n_filters, 5, 2, 2),
        ConvBlock(n_filters, 2 * n_filters, 5, 2, 2),
        ConvBlock(2 * n_filters, 4 * n_filters, 3, 2, 1),
        ConvBlock(4 * n_filters, 6 * n_filters, 3, 2, 1),
        nn.Flatten(),
        nn.Linear(feat * feat * 6 * n_filters, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, n_outputs),
    )


class FaceDecoder(nn.Module):
    def __init__(self, latent_dim=100, n_filters=12, image_size=64):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_filters = n_filters
        self.feat_size = image_size // 16
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, self.feat_size * self.feat_size * 6 * n_filters),
            nn.ReLU(inplace=True),
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(6 * n_filters, 4 * n_filters, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * n_filters, n_filters, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_filters, 3, 5, 2, 2, output_padding=1),
        )

    def forward(self, z):
        x = self.linear(z).view(-1, 6 * self.n_filters, self.feat_size, self.feat_size)
        return self.deconv(x)


class StandardCNN(nn.Module):
    def __init__(self, image_size=64, n_filters=12):
        super().__init__()
        self.net = build_encoder(3, 1, image_size, n_filters)

    def forward(self, x):
        return self.net(x)

    @torch.inference_mode()
    def predict(self, x):
        was_training = self.training
        self.eval()
        try:
            return self.net(x)
        finally:
            if was_training:
                self.train()


def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    return mean + std * torch.randn_like(std)


@dataclass
class DBVAEOutput:
    y_logit: torch.Tensor
    z_mean: torch.Tensor
    z_logvar: torch.Tensor
    z: torch.Tensor
    x_recon: torch.Tensor


class DBVAE(nn.Module):
    def __init__(self, latent_dim=100, image_size=64, n_filters=12):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = build_encoder(3, 1 + 2 * latent_dim, image_size, n_filters)
        self.decoder = FaceDecoder(latent_dim, n_filters, image_size)

    def encode(self, x):
        h = self.encoder(x)
        return h[:, 0:1], h[:, 1:1 + self.latent_dim], h[:, 1 + self.latent_dim:]

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        y_logit, z_mean, z_logvar = self.encode(x)
        z = reparameterize(z_mean, z_logvar)
        return DBVAEOutput(y_logit, z_mean, z_logvar, z, self.decoder(z))

    @torch.inference_mode()
    def predict(self, x):
        was_training = self.training
        self.eval()
        try:
            return self.encode(x)[0]
        finally:
            if was_training:
                self.train()

    @torch.inference_mode()
    def latent_mean(self, x):
        was_training = self.training
        self.eval()
        try:
            return self.encode(x)[1]
        finally:
            if was_training:
                self.train()
