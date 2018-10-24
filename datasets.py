import numpy as np
from torch.utils.data import Dataset
import torchvision
import os
import PIL.Image

class MNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        return super().__getitem__(index)[0]

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        return super().__getitem__(index)[0]

class CELEBA(torchvision.datasets.ImageFolder):
    def __init__(self, root='./data/celeba/', train=True, transform=None):   
        if train:
            root = root + 'train'
        else:
            root = root + 'test'
        super().__init__(root=root, transform=transform)
    
    def __getitem__(self, index):
        return super().__getitem__(index)[0]