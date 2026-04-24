# Author: proud cheetah group

import torch
import torch.nn as nn
import torch.nn.functional as F


class SongAutoencoder(nn.Module):
    def __init__(self, input_dim=6, latent_dim=128):
        super().__init__()
        # 6 audio features -> 128
        # The input features expected: danceability, energy, valence, acousticness, speechiness, tempo

        # Encoder maps compact numeric audio features into the same latent
        # dimension used by the text transformer embeddings.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, latent_dim),
        )

        # Decoder reconstructs the original audio features, giving the latent
        # space a self-supervised signal in addition to contrastive matching.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, input_dim),
            nn.Sigmoid(),  # Sigmoid because inputs are Min-Max scaled to [0,1]
        )

    def forward(self, x):
        # Return both the retrieval embedding and reconstruction so train.py can
        # optimize contrastive and reconstruction losses together.
        z = self.encode(x)
        reconstruction = self.decoder(z)
        return z, reconstruction

    def encode(self, x):
        latent = self.encoder(x)
        # L2 normalize so we can do Cosine Similarity / InfoNCE
        latent_normalized = F.normalize(latent, p=2, dim=1)
        return latent_normalized
