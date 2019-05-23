import os
from skimage import io
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.file_name = sorted(os.listdir(root))
        self.len = len(self.file_name)
        self.transform = transform
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        image = io.imread(os.path.join(self.root, self.file_name[index]))

        if self.transform:
            image = self.transform(image)
        
        return image
