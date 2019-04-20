import os
import sys
from skimage import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import MyDataset


def inv_normalize(img):
    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406

    img = img.transpose(1,2,0)
    return img


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Preprocessing for ImageNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2], exist_ok=True)

    dataset = MyDataset.ImageDataset(img_dir=sys.argv[1], label_path="./labels.npy", transform=transform)
    loader = DataLoader(dataset)

    model = models.resnet50(pretrained=True)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    EPSILON = 0.1

    for i, data in enumerate(loader):
        img, label = data
        img, label = img.to(device), label.to(device)
        img.requires_grad = True

        output = model(img)
        
        loss = criterion(output, label)
        loss.backward()

        sign_grad = img.grad.sign()
        noise = EPSILON * sign_grad
        atk_img = img + noise
        
        atk_img = inv_normalize(atk_img.squeeze(0).detach().cpu().numpy())
        atk_img = np.clip(atk_img, 0.0, 1.0) * 255
        atk_img = atk_img.astype(np.uint8)

        io.imsave(os.path.join(sys.argv[2], "{}.png".format(str(i).zfill(3))), atk_img)
        
        print("Finish image {}".format(str(i).zfill(3)))