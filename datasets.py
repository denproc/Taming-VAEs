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
    
class CELEBA(Dataset):
    def __init__(self, root='./data/celeba/img_align_celeba/', transform=None):
        super().__init__()
        self.transform = transform
        self.len = 10000
        try:
            files = os.listdir(root)
        except OSError():
            raise OSError('Specify the directory with CELEBA dataset')
        files.sort()
        files = files[:self.len]
        self.data = []
        for file in files:
            self.data.append(np.asarray(PIL.Image.open(root+file))[None,...])
        self.data = np.concatenate(self.data)
        print (self.data.shape)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index])
        else:
            return self.data[index]