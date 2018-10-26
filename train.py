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

def RE(x, mu, tol):
    return torch.sum(torch.pow(mu - x, 2), dim = 1) - tol**2

def RE_mtr(x, mu, tol):
    return torch.sum(torch.pow(mu - x, 2), dim = (1, 2)) - tol**2

def loss_beta_vae(x, mu_gen, logsigma_gen, mu_latent, logsigma_latent, beta=1):
    return torch.mean(beta * KL_divergence(mu_latent, logsigma_latent) - log_likelihood(x, mu_gen, logsigma_gen))



def draw_hist(train_hist, valid_hist, lambd_hist = [0]):
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

    
def train_geco_draw(model, opt, scheduler, train_loader, valid_loader, 
                   lambd_init = torch.FloatTensor([1]), 
                   constraint_f = RE_mtr, num_epochs=20, 
                   lbd_step = 100, alpha = 0.99, visualize = True, 
                   device = 'cpu', tol = 1, pretrain = 1):
    
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
        
        for X_batch in train_loader:
            X_batch = X_batch.to(device)
            
            reconstruction_mu, KL = model(X_batch)
            constraint = torch.mean(constraint_f(X_batch, reconstruction_mu, tol = tol))
            KL_div = torch.mean(KL.sum(dim = (1,2,3)))
            loss = KL_div + lambd * constraint
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                if epoch == 0 and iter_num == 0:
                    constrain_ma = constraint
                else:
                    constrain_ma = alpha * constrain_ma.detach_() + (1 - alpha) * constraint
                if iter_num % lbd_step == 0 and epoch > pretrain:
                    lambd *= torch.clamp(torch.exp(constrain_ma), 0.9, 1.05)
                                        
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
            for X_batch in valid_loader:
                X_batch = X_batch.to(device)
                reconstruction_mu, KL = model(X_batch)
                constraint = torch.mean(constraint_f(X_batch, reconstruction_mu, tol = tol))
                KL_div = torch.mean(KL.sum(dim = (1,2,3)))
                loss = KL_div + lambd * constraint
                
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
            draw_hist(train_hist, valid_hist, lambd_hist)

        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_hist['loss'][-1]))
        print("  validation loss (in-iteration): \t{:.6f}".format(valid_hist['loss'][-1]))

    

def train_geco(model, opt, scheduler, train_loader, valid_loader, 
               lambd_init = torch.FloatTensor([1]), 
               KL_divergence = KL_divergence, 
               constraint_f = RE, num_epochs=20, 
               lbd_step = 100, alpha = 0.99, visualize = True, 
               device = 'cpu', tol = 1, pretrain = 1):
    
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
        
        for X_batch in train_loader:
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
                if iter_num % lbd_step == 0 and epoch > pretrain:
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
            for X_batch in valid_loader:
                X_batch = X_batch.reshape(train_loader.batch_size, -1).to(device)
                reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = model(X_batch)

                constraint = torch.mean(constraint_f(X_batch, reconstruction_mu, tol = tol))
                KL_div = torch.mean(KL_divergence(latent_mu, latent_logsigma))
                loss = KL_div + lambd * constraint
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
            draw_hist(train_hist, valid_hist, lambd_hist)

        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_hist['loss'][-1]))
        print("  validation loss (in-iteration): \t{:.6f}".format(valid_hist['loss'][-1]))
        
        
def train_beta(model, opt, scheduler, train_loader, valid_loader, device = 'cuda', 
               loss_beta_vae=loss_beta_vae, num_epochs=20, beta=1):
    
    train_hist = {'loss':[], 'reconstr':[], 'KL':[]}
    valid_hist = {'loss':[], 'reconstr':[], 'KL':[]}
    model.to(device)
    for epoch in range(num_epochs):
        # a full pass over the training data:
        start_time = time.time()
        model.train(True)
        train_hist['loss'].append(0)
        train_hist['reconstr'].append(0)
        train_hist['KL'].append(0)
    
        for X_batch in train_loader:
            X_batch = X_batch.reshape(train_loader.batch_size, -1).to(device)
            reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = model.forward(X_batch)
            loss = loss_beta_vae(X_batch, reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, beta)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_hist['loss'][-1] += loss.data.cpu().numpy()/len(train_loader)

        # a full pass over the validation data:
        model.train(False)
        valid_hist['loss'].append(0)
        valid_hist['reconstr'].append(0)
        valid_hist['KL'].append(0)
        with torch.no_grad():
            for X_batch in valid_loader:
                X_batch = X_batch.reshape(valid_loader.batch_size, -1).to(device)

                reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = model.forward(X_batch)
                loss = loss_beta_vae(X_batch, reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma, beta)
                valid_hist['loss'][-1] += loss.data.cpu().numpy()/len(valid_loader)
               
        
        # update lr
        if scheduler is not None:
            scheduler.step(valid_hist['loss'][-1])
        # stop
        if opt.param_groups[0]['lr'] <= 1e-6:
            break
        
        # visualization of training
        display.clear_output(wait=True)
        draw_hist(train_hist, valid_hist)

        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_hist['loss'][-1]))
        print("  validation loss (in-iteration): \t{:.6f}".format(valid_hist['loss'][-1]))


def train_beta_draw(model, opt, scheduler, train_loader, valid_loader, device = 'cuda',num_epochs=20, beta=1):
   
    train_hist = {'loss':[], 'reconstr':[], 'KL':[]}
    valid_hist = {'loss':[], 'reconstr':[], 'KL':[]}
 
    model.to(device)
    for epoch in range(num_epochs):
        # a full pass over the training data:
        start_time = time.time()
        train_hist['loss'].append(0)
        train_hist['reconstr'].append(0)
        train_hist['KL'].append(0)
        model.train(True)
        for X_batch in train_loader:
            X_batch = X_batch.to(device)
            reconstruction_mu, KL = model(X_batch)
            KL =  torch.mean(KL.sum(dim = (1,2,3)))
            reconstr = torch.mean(RE_mtr(reconstruction_mu, X_batch, 0))
            loss = reconstr + beta * KL
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_hist['loss'][-1] += loss.data.cpu().numpy()/len(train_loader)


        # a full pass over the validation data:
        model.train(False)
        valid_hist['loss'].append(0)
        valid_hist['reconstr'].append(0)
        valid_hist['KL'].append(0)
        with torch.no_grad():
            for X_batch in train_loader:
                X_batch = X_batch.to(device)
                reconstruction_mu, KL = model(X_batch)
                KL =  torch.mean(KL.sum(dim = (1,2,3)))
                reconstr = torch.mean(RE_mtr(X_batch,reconstruction_mu, 0))
                loss = reconstr + beta * KL
                valid_hist['loss'][-1] += loss.data.cpu().numpy()/len(valid_loader)
                
        
        # update lr
        if scheduler is not None:
            scheduler.step(valid_hist['loss'][-1])
        # stop
        if opt.param_groups[0]['lr'] <= 1e-6:
            break
        
        # visualization of training
        display.clear_output(wait=True)
        draw_hist(train_hist, valid_hist)

        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_hist['loss'][-1] ))
        print("  validation loss (in-iteration): \t{:.6f}".format(valid_hist['loss'][-1]))
        
     
