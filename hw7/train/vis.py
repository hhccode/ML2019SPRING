import numpy as np
from sklearn.cluster import KMeans
from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import visdom
from models import AutoEncoder

class VisDataset(Dataset):
    def __init__(self, root, transform):
        self.data = np.load(root)
        self.transform = transform
        self.len = len(self.data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.transform:
            image = self.transform(self.data[index])
        
        return image

def main():
    vis = visdom.Visdom()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    batch_size = 256
    latent_dim = 64

    dataset = VisDataset("./visualization.npy", transform=transform)
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
    tsne = TSNE(n_components=2, perplexity=50.0, n_iter=3000, verbose=1, random_state=0, n_jobs=20)
    X = tsne.fit_transform(embeddings)
    X_min, X_max = X.min(0), X.max(0)
    X_tsne = (X - X_min) / (X_max - X_min)

    kmeans = KMeans(n_clusters=2, max_iter=3000, random_state=0)
    kmeans.fit(X_tsne)

    gt = np.concatenate((np.ones(2500, dtype=np.int), np.zeros(2500, dtype=np.int)))
    
    vis.scatter(X=X_tsne, Y=kmeans.labels_+1, win="Prediction", 
                opts={  
                    "legend": ["A", "B"],
                    "markersize": 5,
                    "title": "Predictied distribution"
                })
    vis.scatter(X=X_tsne, Y=gt+1, win="Ground Truth",
                opts={  
                    "legend": ["A", "B"],
                    "markersize": 5,
                    "title": "GT distribution"
                })


if __name__ == "__main__":
    main()