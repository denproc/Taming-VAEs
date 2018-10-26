import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch

dims = [6075, 1024, 512]
dim_latent = 100

class VAE(nn.Module):
    def __init__(self, dims = dims, dim_latent=dim_latent):
        super().__init__()
        
        self.encode = nn.Sequential()
        curr_l = dims[0]
        for i, next_layer in enumerate(dims[1:]):
            self.encode.add_module('Linear_{}'.format(i+1), nn.Linear(curr_l, next_layer))
            self.encode.add_module('ReLU_{}'.format(i+1),nn.ReLU())
            curr_l = next_layer
            
        self.decode = nn.Sequential()
        curr_l = dim_latent
        for i, next_layer in enumerate(dims[:0:-1]):
            self.decode.add_module('Linear_{}'.format(i+1), nn.Linear(curr_l, next_layer))
            self.decode.add_module('ReLU_{}'.format(i+1), nn.ReLU())
            curr_l = next_layer

        
        self.latent_mu = nn.Linear(dims[-1], dim_latent)
        self.latent_logsigma = nn.Linear(dims[-1], dim_latent)
        
    
        self.reconstruction_mu = nn.Sequential(
            nn.Linear(dims[1], dims[0]),
            nn.Sigmoid()
        )
        
        self.reconstruction_logsigma = nn.Sequential(
            nn.Linear(dims[1], dims[0]),
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
        
        return reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, z

    
class IVAE(nn.Module):
    def __init__(self, dims = dims, dim_latent=dim_latent):
        super().__init__()
        self.num_samples = 5
        self.encode = nn.Sequential()
        curr_l = dims[0]
        for i, next_layer in enumerate(dims[1:]):
            self.encode.add_module('Linear_{}'.format(i+1), nn.Linear(curr_l, next_layer))
            self.encode.add_module('ReLU_{}'.format(i+1),nn.ReLU())
            curr_l = next_layer
            
        self.decode = nn.Sequential()
        curr_l = dim_latent
        for i, next_layer in enumerate(dims[:0:-1]):
            self.decode.add_module('Linear_{}'.format(i+1), nn.Linear(curr_l, next_layer))
            self.decode.add_module('ReLU_{}'.format(i+1), nn.ReLU())
            curr_l = next_layer

        
        self.latent_mu = nn.Linear(dims[-1], dim_latent)
        self.latent_logsigma = nn.Linear(dims[-1], dim_latent)
        
    
        self.reconstruction_mu = nn.Sequential(
            nn.Linear(dims[1], dims[0]),
            nn.Sigmoid()
        )
        
        self.reconstruction_logsigma = nn.Sequential(
            nn.Linear(dims[1], dims[0]),
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
        latent_mu = self.latent_mu(x_enc).view(x.size(0), 1, -1).repeat(1,self.num_samples,1)
        latent_logsigma = self.latent_logsigma(x_enc).view(x.size(0), 1, -1).repeat(1,self.num_samples,1)
        z = self.gaussian_sampler(latent_mu, latent_logsigma)
        
        x_hat = self.decode(z)
        
        reconstruction_mu = self.reconstruction_mu(x_hat)
        reconstruction_logsigma = self.reconstruction_logsigma(x_hat)
        
        return reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, z