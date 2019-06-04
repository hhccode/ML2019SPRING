import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import ImageDataset
from models import MobileNetV1
# import visdom

def readfile(path):
    print("Reading File...")
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
        if (i % 10 == 0):
            x_val.append(tmp)
            y_val.append(raw_train[i][0])
        else:
            x_train.append(tmp)
            y_train.append(raw_train[i][0])

    x_train = np.array(x_train, dtype=float) / 255.0
    x_val = np.array(x_val, dtype=float) / 255.0
    y_train = np.array(y_train, dtype=int)
    y_val = np.array(y_val, dtype=int)
    x_train = torch.FloatTensor(x_train)
    x_val = torch.FloatTensor(x_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)

    return x_train, y_train, x_val, y_val

def main(argv):
    #vis = visdom.Visdom()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    x_train, y_train, x_val, y_val = readfile(argv[1])

    train_trans = [
        T.Compose([
            T.ToPILImage(),
            T.RandomRotation(30),
            T.RandomCrop(40),
            T.Resize(48),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
    ]

    val_trans = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])

    train_dataset = ImageDataset(x=x_train, y=y_train, transform=val_trans)
    val_dataset = ImageDataset(x=x_val, y=y_val, transform=val_trans)
    
    for tran in train_trans:
        train_dataset = train_dataset.__add__(ImageDataset(x=x_train, y=y_train, transform=tran))

    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=True)
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    model = MobileNetV1()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    EPOCH = 300
    best_acc = 0.0

    for epoch in range(EPOCH):
        
        train_loss, train_acc = 0.0, 0.0
        model.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predict = torch.max(outputs, 1)[1]
            acc = np.mean((labels == predict).cpu().numpy())

            train_loss += loss.item()
            train_acc += acc

            print("\rEpoch: [{}/{}] | Step: [{}/{}] | \
                Acc: {:.5f} | Loss: {:.5f}".format(epoch+1, EPOCH,
                                                    i+1, train_steps,
                                                    acc, loss.item()), end="")
        
        val_loss, val_acc = 0.0, 0.0
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                predict = torch.max(outputs, 1)[1]

                val_loss += loss.item()
                val_acc += np.mean((labels == predict).cpu().numpy())
        
        
        # vis.line(
        #     X=np.array([epoch+1]),
        #     Y=np.array([train_loss/train_steps, val_loss/val_steps]).reshape(1, 2),
        #     win="Loss", update="append", 
        #     opts={  
        #         "legend": ["train loss", "val loss"],
        #         "xlabel": "Epoch",
        #         "ylabel": "Loss",
        #         "title": "Loss curve"
        #     }
        # )

        # vis.line(
        #     X=np.array([epoch+1]),
        #     Y=np.array([train_acc/train_steps, val_acc/val_steps]).reshape(1, 2),
        #     win="Acc", update="append", 
        #     opts={  
        #         "legend": ["train acc", "val acc"],
        #         "xlabel": "Epoch",
        #         "ylabel": "Accuracy",
        #         "title": "Accuracy curve"
        #     }
        # )

        if val_acc / val_steps > best_acc:
            best_acc = val_acc / val_steps
            torch.save(model.half().state_dict(), "./checkpoints/{}_{:.5f}.ckpt".format(epoch+1, best_acc), pickle_protocol=pickle.HIGHEST_PROTOCOL)
            model.float()

if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)
    main(sys.argv)