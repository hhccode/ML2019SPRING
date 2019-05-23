import os
import multiprocessing as mp
import numpy as np
from sklearn.decomposition import PCA
from skimage import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import ImageDataset
from models import AutoEncoder

def autoencoder():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    batch_size = 256
    latent_dim = 64

    dataset = ImageDataset("./images", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size)

    model = AutoEncoder(enc_channel=8, dec_channel=32, latent_dim=latent_dim)
    model.to(device)
    model.load_state_dict(torch.load("./model/weights.ckpt", map_location=device))
    model.eval()

    criterion = nn.MSELoss()
    loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data.to(device)

            outputs = model(inputs)
            loss += criterion(outputs, inputs).item()
    

    loss /= len(loader)

    print("Autoencoder: Reconstruction loss = {}".format(loss))

def read_images(path):
    return io.imread(path).astype(np.float32).flatten()

def pca():
    files = [os.path.join("./images", file) for file in os.listdir("./images")]
    
    pool = mp.Pool(8)
    X = np.array(pool.map(read_images, files))
    pool.close()
    pool.join()
    
    X /= 255.0
    X = (X - 0.5) / 0.5

    mean = np.mean(X, axis=0)

    pca = PCA(n_components=10, random_state=0)
    X_reduced = pca.fit_transform(X-mean)

    X_recon = mean + np.dot(X_reduced, pca.components_)

    X_loss = ((X - X_recon) ** 2).mean(axis=1)
    loss = X_loss.mean()

    print("PCA: Reconstruction loss = {}".format(loss))

if __name__ == "__main__":
    autoencoder()
    pca()