import sys
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import ImageDataset
from models import AutoEncoder

def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    batch_size = 256
    latent_dim = 64

    dataset = ImageDataset(argv[1], transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size)

    model = AutoEncoder(enc_channel=8, dec_channel=32, latent_dim=latent_dim)
    model.to(device)
    model.load_state_dict(torch.load("./model/weights.ckpt", map_location=device))
    model.eval()

    embeddings = np.array([]).reshape(0, latent_dim)
    with torch.no_grad():
        for i, data in enumerate(loader):
            print(i)
            inputs = data.to(device)

            outputs = model.enc_fc(model.encoder(inputs).contiguous().view(inputs.size(0), -1)).cpu().numpy()
            embeddings = np.concatenate((embeddings, outputs))
    
    # TSNE
    tsne = TSNE(n_components=2, verbose=1, random_state=0, n_jobs=20)
    X = tsne.fit_transform(embeddings)
    X_min, X_max = X.min(0), X.max(0)
    X_tsne = (X - X_min) / (X_max - X_min)

    kmeans = KMeans(n_clusters=2, max_iter=3000, random_state=0)
    kmeans.fit(X_tsne)

    df = pd.read_csv(argv[2])
    IDs, name1, name2 = np.array(df['id']), np.array(df['image1_name']), np.array(df['image2_name'])

    with open(argv[3], "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label"])

        for id, n1, n2 in zip(IDs, name1, name2):
            if kmeans.labels_[n1-1] == kmeans.labels_[n2-1]:
                writer.writerow([id, 1])
            else:
                writer.writerow([id, 0])


if __name__ == "__main__":
    main(sys.argv)