import sys
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import MyDataset
import MyNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #random.seed(0)
    #torch.manual_seed(0)
    #torch.cuda.manual_seed(0)
    device = torch.device('cuda')
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
    # Original data
    dataset = MyDataset.ImageDataset(file_path="./train.csv", transform=transforms.ToTensor())
    
    trans = [transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(48),
                transforms.ToTensor()  # Convert the numpy array to tensor and normalize to [0, 1]
            ]),
             transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(1.0),
                transforms.ToTensor()  # Convert the numpy array to tensor and normalize to [0, 1]
            ]),
             transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(48),
                transforms.RandomHorizontalFlip(1.0),
                transforms.ToTensor()  # Convert the numpy array to tensor and normalize to [0, 1]
            ]),
             transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(30),
                transforms.ToTensor()
             ])]

    # Data augmentation
    for tran in trans:
        dataset = dataset.__add__(MyDataset.ImageDataset(file_path="./train.csv", transform=tran))

    size = len(dataset)
    train_size = int(size * 0.8)
    valid_size = size - train_size

    # Divide the data into training data and validation data
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4)


    model = MyNet.ImageNet()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)


    TRAIN_LOSS, TRAIN_ACC = [], []
    VALID_LOSS, VALID_ACC = [], []
    EPOCH = 250
    best_acc = 0.0

    for epoch in range(EPOCH):
        
        train_loss, train_acc = 0.0, 0.0
        model.train()

        for i, data in enumerate(train_loader):
            print(epoch, i)
            # Get the mini-batch
            # Inputs: (batch_size, 1, 48, 48)
            # Labels: (batch_size)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predict = torch.max(outputs, 1)[1]

            train_loss += loss.item()
            train_acc += np.mean((labels == predict).cpu().numpy())

        TRAIN_LOSS.append(train_loss / len(train_loader))
        TRAIN_ACC.append(train_acc / len(train_loader))
        
        valid_loss, valid_acc = 0.0, 0.0
        model.eval()

        with torch.no_grad():
            if epoch != EPOCH-1:
                for i, data in enumerate(valid_loader):
                    print(epoch, i)
                    # Get the mini-batch
                    # Inputs: (batch_size, 1, 48, 48)
                    # Labels: (batch_size)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)


                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    predict = torch.max(outputs, 1)[1]

                    valid_loss += loss.item()
                    valid_acc += np.mean((labels == predict).cpu().numpy())
            else:
                for i, data in enumerate(valid_loader):
                    print(epoch, i)
                    # Get the mini-batch
                    # Inputs: (batch_size, 1, 48, 48)
                    # Labels: (batch_size)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)


                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if i == 0:
                        y_pred = torch.max(outputs, 1)[1].cpu().numpy()
                        y_true = labels.cpu().numpy()
                    else:
                        y_pred = np.append(y_pred, torch.max(outputs, 1)[1].cpu().numpy())
                        y_true = np.append(y_true, labels.cpu().numpy())
                

                    predict = torch.max(outputs, 1)[1]

                    valid_loss += loss.item()
                    valid_acc += np.mean((labels == predict).cpu().numpy())
        
                cm = confusion_matrix(y_true, y_pred).astype('float')

        VALID_LOSS.append(valid_loss / len(valid_loader))
        VALID_ACC.append(valid_acc / len(valid_loader))
        
        if (valid_acc / len(valid_loader)) > best_acc:
            best_acc = valid_acc / len(valid_loader)
            torch.save(model, "./model/model-epoch{}-train_acc{:.5f}-valid_acc{:.5f}.pkl".format(epoch, train_acc / len(train_loader), valid_acc / len(valid_loader)))

    EPOCH = [i+1 for i in range(EPOCH)]
    
    plt.figure(1)
    plt.plot(EPOCH, TRAIN_LOSS, color="r", label="train loss")
    plt.plot(EPOCH, VALID_LOSS, color="b", label="valid loss")
    plt.legend(loc='upper right')

    plt.figure(2)
    plt.plot(EPOCH, TRAIN_ACC, color="r", label="train acc")
    plt.plot(EPOCH, VALID_ACC, color="b", label="valid acc")
    plt.legend(loc='lower right')
    plt.show()


    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(3)
    plt.imshow(cm, interpolation='nearest', cmap='jet')
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
 
    row, col = cm.shape
    for r in range(row):
        for c in range(col):
            plt.text(c, r, "{:.3f}".format(cm[r][c]), horizontalalignment='center', verticalalignment='center', color='white')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.show()
