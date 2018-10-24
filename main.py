import argparse
import torch
import torchvision
import datasets

batch_size = 4

if (__name__=='__main__'):
    parser = argparse.ArgumentParser(description='Taming VAEs experiments')
    parser.add_argument('--data', dest='dataset', default=None, help='Dataset to be used')
    args = parser.parse_args()
    
    if (args.dataset.lower() == 'mnist'):
        data_set = datasets.MNIST('./data/mnist/', download=True, transform=torchvision.transforms.ToTensor())
    elif (args.dataset.lower() == 'cifar10'):
        data_set = datasets.CIFAR10('./data/cifar10/', download=True, transform=torchvision.transforms.ToTensor())
    elif (args.dataset.lower() == 'celeba'):
        data_set = datasets.CELEBA('./data/celeba/img_align_celeba/', transform=torchvision.transforms.ToTensor())
        
    loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch in loader:
        print (batch.size())
        break