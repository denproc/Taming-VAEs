import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
import time
from IPython import display
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook


def KL_divergence(mu, logsigma):
    return - 0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - logsigma.exp().pow(2), dim=1)

def RE(x, mu, tol):
    return torch.sum(torch.pow(mu - x, 2), dim = 1) - tol**2


def train_geco(model, opt, scheduler, train_loader, valid_loader, 
               lambd_init = torch.FloatTensor([1]), 
               KL_divergence = KL_divergence, 
               constraint_f = RE, num_epochs=20, 
               lbd_step = 100, alpha = 0.99, visualize = True, 
               device = 'cpu', tol = 1):
    
    model.to(device)
    
    train_hist = {'loss':[], 'reconstr':[], 'KL':[]}
    valid_hist = {'loss':[], 'reconstr':[], 'KL':[]}
    lambd_hist = []
    
    lambd = lambd_init.to(device)
    iter_num = 0
    
    for epoch in range(num_epochs):
        # a full pass over the training data:
        start_time = time.time()
        
        model.train(True)
        train_hist['loss'].append(0)
        train_hist['reconstr'].append(0)
        train_hist['KL'].append(0)
        
        for (X_batch, y_batch) in train_loader:
            X_batch = X_batch.reshape(train_loader.batch_size, -1).to(device)
            
            reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = model(X_batch)
            constraint = torch.mean(constraint_f(X_batch, reconstruction_mu, tol = tol))
            KL_div = torch.mean(KL_divergence(latent_mu, latent_logsigma))
            loss = KL_div + lambd * constraint
#             loss.backward(retain_graph = True)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                if epoch == 0 and iter_num == 0:
                    constrain_ma = constraint
                else:
                    constrain_ma = alpha * constrain_ma.detach_() + (1 - alpha) * constraint
                if iter_num % lbd_step == 0:
#                     print(torch.exp(constrain_ma), lambd)
                    lambd *= torch.clamp(torch.exp(constrain_ma), 0.9, 1.1)
                                        
            train_hist['loss'][-1] += loss.data.cpu().numpy()[0]/len(train_loader)
            train_hist['reconstr'][-1] += constraint.data.cpu().numpy()/len(train_loader)
            train_hist['KL'][-1] += KL_div.data.cpu().numpy()/len(train_loader)                                        
            iter_num += 1
        lambd_hist.append(lambd.data.cpu().numpy()[0]) 

           
        model.train(False)
        valid_hist['loss'].append(0)
        valid_hist['reconstr'].append(0)
        valid_hist['KL'].append(0)
        with torch.no_grad():
            for (X_batch, y_batch) in valid_loader:
                X_batch = X_batch.reshape(train_loader.batch_size, -1).to(device)
                reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = model(X_batch)

                constraint = torch.mean(constraint_f(X_batch, reconstruction_mu, tol = tol))
                KL_div = torch.mean(KL_divergence(latent_mu, latent_logsigma))

                valid_hist['loss'][-1] += loss.data.cpu().numpy()[0]/len(valid_loader)
                valid_hist['reconstr'][-1] += constraint.data.cpu().numpy()/len(valid_loader)
                valid_hist['KL'][-1] += KL_div.data.cpu().numpy()/len(valid_loader)
        

        # update lr
        if scheduler is not None:
            scheduler.step(valid_hist['loss'][-1])
        # stop
        if opt.param_groups[0]['lr'] <= 1e-6:
            break
        
        
        # visualization of training
        if visualize:
            display.clear_output(wait=True)
            fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))

            ax[0].plot(train_hist['loss'], label = 'train')
            ax[0].plot(valid_hist['loss'], label = 'test')
            ax[0].legend()
            ax[0].set_title("Total Loss")

            ax[1].plot(train_hist['reconstr'], label = 'train')
            ax[1].plot(valid_hist['reconstr'], label = 'test')
            ax[1].legend()
            ax[1].set_title("Reconstruction")

            ax[2].plot(train_hist['KL'], label = 'train')
            ax[2].plot(valid_hist['KL'], label = 'test')
            ax[2].legend()
            ax[2].set_title("KL divergence")
            
            ax[3].plot(lambd_hist)
            ax[3].set_title("Lambda")
            plt.show()


        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_hist['loss'][-1]))
        print("  validation loss (in-iteration): \t{:.6f}".format(valid_hist['loss'][-1]))
