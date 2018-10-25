import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch

class VAE(nn.Module):
    def __init__(self, dim_init=6075, dim_middle=1024, dim_latent=100):
        super(VAE, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(dim_init, dim_middle),
            nn.ReLU(),
            nn.Linear(dim_middle, dim_middle),
            nn.ReLU()
        )
        
        self.latent_mu = nn.Linear(dim_middle, dim_latent)
        self.latent_logsigma = nn.Linear(dim_middle, dim_latent)
        
        self.decode = nn.Sequential(
            nn.Linear(dim_latent, dim_middle),
            nn.ReLU(),
            nn.Linear(dim_middle, dim_middle),
            nn.ReLU()
        )

        self.reconstruction_mu = nn.Sequential(
            nn.Linear(dim_middle, dim_init),
            nn.Sigmoid()
        )
        
        self.reconstruction_logsigma = nn.Sequential(
            nn.Linear(dim_middle, dim_init),
            nn.Sigmoid()
        )
        
    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = logsigma.exp()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        
        x_enc = self.encode(x)
        latent_mu = self.latent_mu(x_enc)
        latent_logsigma = self.latent_logsigma(x_enc)
        
        z = self.gaussian_sampler(latent_mu, latent_logsigma)
        
        x_hat = self.decode(z)
        
        reconstruction_mu = self.reconstruction_mu(x_hat)
        reconstruction_logsigma = self.reconstruction_logsigma(x_hat)
        
        return reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma
