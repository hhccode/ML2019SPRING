import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.x[index]
        image = self.transform(image)

        label = self.y[index]
        
        return image, label