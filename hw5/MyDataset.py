import os
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, img_dir, label_path, transform):
        self.img_dir = img_dir
        self.len = len([fname for fname in os.listdir(img_dir) if ".png" in fname])
        self.transform = transform
        self.labels = np.load(label_path)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name = "{}.png".format(str(index).zfill(3))
        img_path = os.path.join(self.img_dir, img_name)
        
        img = io.imread(img_path)
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[index]