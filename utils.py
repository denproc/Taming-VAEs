from torch.distributions import Normal
import torch
from train import KL_divergence, RE


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def sample_vae(hid_size, model):
    z = Normal(torch.zeros(hid_size), torch.zeros(hid_size)+1).sample()
    mu = model.reconstruction_mu(model.decode(z.to(device)))
    return mu

def marginal_KL(valid_loader, model):
    KL = 0
    for X_batch in valid_loader:
        X_batch = X_batch.reshape(valid_loader.batch_size, -1).to(device)
        rec_mu, rec_logsigma, latent_mu, latent_logsigma = model(X_batch)
        KL += torch.sum(KL_divergence(latent_mu, latent_logsigma))
    return KL/(len(valid_loader)*valid_loader.batch_size)

def Compute_NLL(valid_loader, model):
    rec_loss = 0
    for X_batch in valid_loader:
        X_batch = X_batch.reshape(valid_loader.batch_size, -1).to(device)
        rec_mu, rec_logsigma, latent_mu, latent_logsigma = model(X_batch)
        rec_loss += RE(X_batch, rec_mu, 0)
    return rec_loss/(len(valid_loader)*valid_loader.batch_size)       
    
              
              
