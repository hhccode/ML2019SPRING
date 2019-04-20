import sys
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import MyDataset


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    orig = MyDataset.ImageDataset(img_dir="./hw5_data/images", label_path="./labels.npy", transform=None)
    atk = MyDataset.ImageDataset(img_dir=sys.argv[1], label_path="./labels.npy", transform=None)
    loader = DataLoader(atk)
    size = len(orig)

    model = models.resnet50(pretrained=True)
    model.eval()
    
    norm = 0.0
    success = 0

    for i, data in enumerate(loader):
        print(i)
        atk_img, label = data

        atk_img = atk_img.squeeze(0).numpy().astype(int)
        orig_img = orig[i][0].astype(int)

        norm += np.amax(np.abs(atk_img-orig_img))

        atk_img = atk_img.astype(np.uint8)
        atk_img = transform(atk_img).unsqueeze(0)

        output = model(atk_img)
        predict = torch.max(output, 1)[1]
        
        if predict != label:
            success += 1

    print("Norm: {}".format(norm / size))
    print("Success rate: {}".format(success / size))
        
