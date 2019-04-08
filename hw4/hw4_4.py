import numpy as np
import os
import torch
import torch.nn as nn
import copy
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import MyDataset

def cover_image(img, pos, size):
    x, y = pos
    result = copy.deepcopy(img)
    half = size // 2

    for r in range(-half, half+1):
        for c in range(-half, half+1):
            if 0 <= x+r < 48 and 0 <= y+c < 48:
                result[x+r][y+c] = 255
    
    return result


if __name__ == "__main__":
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    model = torch.load("bestmodel.pkl", map_location="cpu")
    model.eval()

    transform = transforms.ToTensor()
    dataset = MyDataset.ImageDataset(file_path="./train.csv", transform=None)
    data = [
        dataset[374],        # 0: angry
        dataset[2275],       # 1: disgust
        dataset[84],         # 2: fear
        dataset[7],          # 3: happy
        dataset[70],         # 4: sad
        dataset[194],        # 5: surprise
        dataset[217]         # 6: neutral
    ]

    KERNEL_SIZE = 15
    sm = nn.Softmax(1)

    for i in range(len(data)):
        x, y = data[i]
        y_true = y.unsqueeze(0)
        heatmap = np.zeros((48,48))
        max_ = 1.0
        pos = None

        for r in range(48):
            for c in range(48):
                x_mask = cover_image(x, (r,c), KERNEL_SIZE)
                x_mask = transform(x_mask).unsqueeze(0)

                outputs = model(x_mask)
                prob = sm(outputs).detach().numpy()[0][i]

                if prob < max_:
                    max_ = prob
                    pos = (r, c)
                
                heatmap[r][c] = prob
        
        plt.figure(i)
        x_mask = cover_image(x, pos, KERNEL_SIZE)
        plt.imshow(x_mask, cmap="gray", norm=NoNorm())
        plt.savefig("mask_img{}.jpg".format(i))
        
        plt.figure(i+1)
        plt.imshow(heatmap, cmap='jet')
        plt.colorbar()
        plt.savefig("heat_{}.jpg".format(i))
        plt.close("all")
        print("Finish saving class {}: {}.".format(i, classes[i]))
        