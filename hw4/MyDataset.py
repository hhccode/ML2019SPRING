import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path).values
        self.transform = transform

    def __to_int(self, x):
        return int(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, image = self.data[index]

        # Convert the list of pixels into int format and reshape as a 48*48 numpy array
        image = np.array(list(map(self.__to_int, image.split())), dtype=np.uint8).reshape(48,48)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label)