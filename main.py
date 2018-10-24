import argparse
import torch
import torchvision

batch_size = 4

if (__name__=='__main__'):
    parser = argparse.ArgumentParser(description='Taming VAEs experiments')
    parser.add_argument('--data', dest='dataset', default=None, help='Dataset to be used')
    args = parser.parse_args()
    
    if (args.dataset.lower() == 'mnist'):
        data_set = torchvision.datasets.MNIST('./data/mnist/', download=True, transform=torchvision.transforms.ToTensor())
    elif (args.dataset.lower() == 'cifar10'):
        data_set = torchvision.datasets.CIFAR10('./data/cifar10/', download=True, transform=torchvision.transforms.ToTensor())
    elif (args.dataset.lower() == 'celeba'):
        data_set = None
        
    loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    for i, _ in loader:
        print (i.size())
        break