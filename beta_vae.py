import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from IPython import display

import argparse
import torchvision

def KL_divergence(mu, logsigma):
    return - 0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - logsigma.exp().pow(2), dim=1)

def log_likelihood(x, mu, logsigma):
    return torch.sum(- logsigma - 0.5 * np.log(2 * np.pi) - (mu - x).pow(2) / (2 * logsigma.exp().pow(2)), dim=1)

def loss_beta_vae(x, mu_gen, logsigma_gen, mu_latent, logsigma_latent, beta=1):
    return torch.mean(beta * KL_divergence(mu_latent, logsigma_latent) - log_likelihood(x, mu_gen, logsigma_gen))



def train_beta(model, opt, scheduler, loss_beta_vae, train_loader, valid_loader, num_epochs=20, beta=1):
    train_loss = []
    valid_loss = []
    
    train_mean_loss = []
    valid_mean_loss = []
    
    for epoch in range(num_epochs):
        # a full pass over the training data:
        start_time = time.time()
        model.train(True)
        for (X_batch, y_batch) in train_loader:
            X_batch = X_batch.reshape(train_loader.batch_size, -1)
            if torch.cuda.is_available():
                reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = \
                model.forward(X_batch.cuda())
                loss = loss_beta_vae(torch.FloatTensor(X_batch).cuda(), \
                                     reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, beta)
                loss.backward()
                opt.step()
                opt.zero_grad()
                train_loss.append(loss.data.cpu().numpy())
            else:
                reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = \
                model.forward(X_batch)
                loss = loss_beta_vae(torch.FloatTensor(X_batch), \
                                     reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, beta)
                loss.backward()
                opt.step()
                opt.zero_grad()
                train_loss.append(loss.data.numpy())

        # a full pass over the validation data:
        model.train(False)
        with torch.no_grad():
            for (X_batch, y_batch) in valid_loader:
                X_batch = X_batch.reshape(valid_loader.batch_size, -1)
                if torch.cuda.is_available():
                    reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = \
                    model.forward(X_batch.cuda())
                    loss = loss_beta_vae(torch.FloatTensor(X_batch).cuda(), reconstruction_mu, \
                                         reconstruction_logsigma, latent_mu, latent_logsigma, beta)
                    valid_loss.append(loss.data.cpu().numpy())
                else:
                    reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = \
                    model.forward(X_batch)
                    loss = loss_beta_vae(torch.FloatTensor(X_batch), reconstruction_mu, \
                                         reconstruction_logsigma, latent_mu, latent_logsigma, beta)
                    valid_loss.append(loss.data.numpy())
                
        train_mean_loss.append(np.mean(train_loss[-len(train_loader) // train_loader.batch_size :]))
        valid_mean_loss.append(np.mean(valid_loss[-len(valid_loader) // valid_loader.batch_size :]))
        
        # update lr
        scheduler.step(valid_mean_loss[-1])
        # stop
        if opt.param_groups[0]['lr'] <= 1e-6:
            break
        
        # visualization of training
        display.clear_output(wait=True)
        plt.figure(figsize=(8, 6))

        plt.title("Loss")
        plt.xlabel("#epoch")
        plt.ylabel("losses")
        plt.plot(train_mean_loss, 'b', label='Training loss')
        plt.plot(valid_mean_loss, 'r', label='Validation loss')
        plt.legend()
        plt.show()

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(
            train_mean_loss[-1]))
        print("  validation loss (in-iteration): \t{:.6f}".format(
            valid_mean_loss[-1]))
