import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from datasets import ImageDataset
from models import AutoEncoder

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    batch_size = 32
    latent_dim = 64

    dataset = ImageDataset("./images", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size)

    model = AutoEncoder(enc_channel=8, dec_channel=32, latent_dim=latent_dim)
    model.to(device)
    model.load_state_dict(torch.load("./model/weights.ckpt", map_location=device))
    model.eval()


    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data.to(device)

            outputs = model(inputs)
            break

    vutils.save_image(inputs.cpu(), "./orig_img.png", normalize=True, range=(-1.0, 1.0))
    vutils.save_image(outputs, "./recon_img.png", normalize=True, range=(-1.0, 1.0))

if __name__ == "__main__":
    main()