import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch

def KL_divergence(mu, logsigma):
    return - 0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - logsigma.exp().pow(2), dim=1)

def RE(x, mu, tol):
    return torch.sum(torch.pow(mu - x, 2)) - tol**2


def train_geco(model, opt, scheduler, train_loader, valid_loader, 
               lambd_init = torch.FloatTensor([1]), 
               KL_divergence = KL_divergence, 
               constraint_f = RE, num_epochs=20, 
               lbd_step = 5, alpha = 0.99, visualize = True, device = 'cuda'):
    
    model.to(device)
    
    train_hist = {'loss':[0], 'reconstr':[0], 'KL':[0]}
    valid_hist = {'loss':[0], 'reconstr':[0], 'KL':[0]}
    
    n_tr_batches = len(train_loader)//train_loader.batch_size
    n_te_natches = len(valid_loader)//valid_loader.batch_size
    
    lambd = lambd_init
    
    for epoch in range(num_epochs):
        # a full pass over the training data:
        start_time = time.time()
        model.train(True)
        for (X_batch, y_batch) in train_loader:
            X_batch.to(device)
    
            reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = model(X_batch)
        
            constraint = constraint_f(X_batch, reconstruction_mu, tol = 0.1)
            KL_div = KL_divergence(latent_mu, latent_logsigma)
            
            if epoch == 0:
                constrain_ma = - constraint
            else:
                constrain_ma = alpha * constrain_ma.detach() - (1 - alpha) * constraint

            loss = KL_div + lambd * constraint
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            if epoch % lbd_step == 0:
                lambd *= torch.exp(constrain_ma) 

            train_hist['loss'][-1] += loss.data.cpu().numpy()[0]
            train_hist['reconstr'][-1] += constraint.data.cpu().numpy()[0]
            train_hist['KL'][-1] += KL_div.data.cpu().numpy()[0]

           
        model.train(False)
        with torch.no_grad():
            for (X_batch, y_batch) in valid_loader:
                X_batch.to(device)
                reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma = model(X_batch)

                constraint = constraint_f(X_batch, reconstruction_mu, tol = 0.1)
                KL_div = KL_divergence(latent_mu, latent_logsigma)

                valid_hist['loss'][-1] += loss.data.cpu().numpy()[0]
                valid_hist['reconstr'][-1] += constraint.data.cpu().numpy()[0]
                valid_hist['KL'][-1] += KL_div.data.cpu().numpy()[0]
        
        train_hist['loss'][-1] /= n_tr_batches
        valid_hist['loss'][-1] /= n_te_batches
        train_hist['reconstr'] /= n_tr_batches
        valid_hist['reconstr'] /= n_te_batches
        train_hist['KL'] /= n_tr_batches
        valid_hist['KL'] /= n_te_batches

        # update lr
        scheduler.step(valid_hist['loss'][-1])
        # stop
        if opt.param_groups[0]['lr'] <= 1e-6:
            break
        
        
        # visualization of training
        if visualize:
            display.clear_output(wait=True)
            fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 5))

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


        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_hist['loss'][-1]))
        print("  validation loss (in-iteration): \t{:.6f}".format(valid_hist['loss'][-1]))