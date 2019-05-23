import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torchvision.transforms as T
import torchvision.utils as vutils
import visdom
from datasets import ImageDataset
from models import AutoEncoder


def main():
    vis = visdom.Visdom()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    transform = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    batch_size = 256
    latent_dim = 64
    EPOCH = 3000

    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./checkpoints/images", exist_ok=True)

    dataset = ImageDataset("./images", transform=transform)
    size = len(dataset)

    train_size = int(size * 0.9)
    val_size = size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_steps = len(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    val_steps = len(val_loader)

    model = AutoEncoder(enc_channel=8, dec_channel=32, latent_dim=latent_dim)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


    for epoch in range(EPOCH):

        model.train()
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs = data.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            print("\rEpoch: [{}/{}], Step: [{}/{}], Loss={:.5f}".format(epoch+1, EPOCH, i+1, train_steps, loss.item()), end="")
            
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0

        for i, data in enumerate(val_loader):
            inputs = data.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            val_loss += loss.item()

        vis.line(
            X=np.array([epoch+1]),
            Y=np.array([train_loss/train_steps, val_loss/val_steps]).reshape(1, 2),
            win="Loss", update="append", 
            opts={  
                "legend": ["train loss", "val loss"],
                "xlabel": "Epoch",
                "ylabel": "loss",
                "title": "Training curve"
            }
        )

        if epoch == 0:
            vutils.save_image(inputs.cpu(), "./checkpoints/images/input.png", normalize=True, range=(-1.0, 1.0))


        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), "./checkpoints/{}.ckpt".format(epoch+1))
            vutils.save_image(outputs, "./checkpoints/images/{}.png".format(epoch+1), normalize=True, range=(-1.0, 1.0))
        

if __name__ == "__main__":
    main()