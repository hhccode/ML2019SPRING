import os
import sys
import csv
from skimage import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import MyDataset
import matplotlib.pyplot as plt

def inv_normalize(img):
    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406

    img = img.transpose(1,2,0)
    return img


if __name__ == "__main__":
    # Preprocessing for ImageNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    category = []
    with open("./hw5_data/categories.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for line in reader: 
            category.append(line[1].replace(", ", "\n"))

    dataset = MyDataset.ImageDataset(img_dir="./hw5_data/images", label_path="./labels.npy", transform=transform)
    data = [
        dataset[50],
        dataset[69],
        dataset[169]
    ]

    model = models.resnet50(pretrained=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    ALPHA = 1
    EPSILON = 0.01
    ITER = 4
    n = [50, 69, 169]
    for i in range(len(data)):
        orig_img, label = data[i]

        orig_img = orig_img.unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        
        orig_output = model(orig_img)
        
        index = torch.argsort(orig_output, dim=1, descending=True)
        sorted_orig_output = orig_output[0, index]
        prob = F.softmax(sorted_orig_output, dim=1)
        prob = prob[0, :3].detach().numpy()
        
        number = [0, 0.5, 1]

        plt.figure(1)
        plt.bar(number, prob, width=0.25, align='center')
        plt.ylabel("Probability")
        plt.title("Original image {}".format(n[i]))
        plt.xticks(number, ("{}({})".format(category[index[0][0]], index[0][0]),
                            "{}({})".format(category[index[0][1]], index[0][1]),
                            "{}({})".format(category[index[0][2]], index[0][2])))
        
        
        img = orig_img
        img.requires_grad = True
        
        for _ in range(ITER):
            output = model(img)
        
            loss = criterion(output, label)
            loss.backward()

            sign_grad = img.grad.sign()
            noise = ALPHA * sign_grad
            atk_img = img + noise
            
            noise = torch.clamp(atk_img-orig_img, min=-EPSILON, max=EPSILON)
            img.data = img + noise
            
            img.grad.data.zero_()
            
        atk_img = inv_normalize(img.squeeze(0).detach().cpu().numpy())
        atk_img = np.clip(atk_img, 0.0, 1.0)

        atk_img = transform(atk_img).unsqueeze(0)
        atk_output = model(atk_img)
        
        index = torch.argsort(atk_output, dim=1, descending=True)
        sorted_atk_output = atk_output[0, index]
        prob = F.softmax(sorted_atk_output, dim=1)
        prob = prob[0, :3].detach().numpy()


        plt.figure(2)
        plt.bar(number, prob, width=0.25, align='center')

        
        plt.ylabel("Probability")
        plt.title("Adversarial image {}".format(n[i]))
        plt.xticks(number, ("{}({})".format(category[index[0][0]], index[0][0]),
                            "{}({})".format(category[index[0][1]], index[0][1]),
                            "{}({})".format(category[index[0][2]], index[0][2])))
        plt.show()
        
        print("Finish image {}".format(str(i).zfill(3)))