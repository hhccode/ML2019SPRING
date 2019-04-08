import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import MyDataset

if __name__ == "__main__":
    device = torch.device("cuda")
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    transform = transforms.ToTensor()
    dataset = MyDataset.ImageDataset(sys.argv[1], transform=None)
    data = [
        dataset[374],        # 0: angry
        dataset[2275],       # 1: disgust
        dataset[84],         # 2: fear
        dataset[7],          # 3: happy
        dataset[70],         # 4: sad
        dataset[194],        # 5: surprise
        dataset[217]         # 6: neutral
    ]

    model = torch.load("./bestmodel.pkl?dl=1")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2], exist_ok=True)

    for i in range(len(data)):
        x, y = data[i]
        orig = x

        x = transform(orig).unsqueeze(0)
        y = y.unsqueeze(0)
        x, y = x.to(device), y.to(device)
        x.requires_grad = True

        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        
        saliency = x.grad.abs().squeeze().cpu().numpy()
        # Normalize to [0, 1]
        saliency = saliency / np.max(saliency)
        
        # Origin
        #plt.figure(1)
        #plt.imshow(orig, cmap='gray', norm=NoNorm())
        #plt.savefig("./origin/fig1_{}.png".format(i))

        # Saliency
        plt.figure(2)
        plt.imshow(saliency, cmap='jet')
        plt.colorbar()
        plt.savefig(os.path.join(sys.argv[2], "fig1_{}.jpg".format(i)))
        plt.close("all")

        print("Finish saving class {}: {}.".format(i, classes[i]))

        #thresh = 0.15
        #mask = orig
        #mask[np.where(saliency < thresh)] = 255

        #plt.figure(3)
        #plt.imshow(mask, cmap='gray', norm=NoNorm())
        #plt.colorbar()
        #plt.savefig("./mask/fig1_{}.png".format(i))
        #plt.close("all")
        
        
