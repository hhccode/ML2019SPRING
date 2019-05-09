import os
import sys
import time
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from models import RNN
from datasets import DcardDataset
#import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists("./ckpt"):
        os.makedirs("./ckpt")


    #dataset = DcardDataset(x_path=sys.argv[1], y_path=sys.argv[2], dict_path=sys.argv[4], corpus_path=(sys.argv[1], sys.argv[3]), w2v_path="./word2vec.model", w2v_pretrain=False)
    
    #dataset = DcardDataset(x_path=sys.argv[1], y_path=sys.argv[2], dict_path=sys.argv[4], w2v_path="./model/word2vec_mincount3.model", w2v_pretrain=True)
    dataset = DcardDataset(x_path=sys.argv[1], y_path=sys.argv[2], dict_path=sys.argv[4], w2v_path="./model/word2vec_mincount5.model", w2v_pretrain=True)
    size = len(dataset)

    train_size = int(size * 0.9)
    val_size = size - train_size

    # Split into train set and val set
    train_dataset, val_dataset = random_split(dataset, (train_size, val_size))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    model = RNN(embedding_dim=128, hidden_dim=256)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(params=model.parameters(), lr=1e-4)

    EPOCH = 30
    TRAIN_LOSS, TRAIN_ACC = [], []
    VAL_LOSS, VAL_ACC = [], []

    for epoch in range(EPOCH):

        train_loss, train_acc = 0.0, 0.0
        model.train()
        start = time.time()

        for i, data in enumerate(train_loader):
            inputs, targets = data
            inputs, targets = inputs.float().to(device), targets.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predict = torch.zeros(inputs.size(0), 1)
                for i in range(inputs.size(0)):
                    if outputs[i, 0] > 0.5:
                        predict[i, 0] = 1.0

                correct = predict.cpu().eq(targets.cpu()).sum()
                acc = float(correct) / inputs.size(0)
                
                train_loss += loss.item()
                train_acc += acc
        
            print("\rEpoch {}, train loss = {:.5f}, train acc = {:.5f}".format(epoch, loss.item(), acc), end="")


        TRAIN_LOSS.append(train_loss / len(train_loader))
        TRAIN_ACC.append(train_acc / len(train_loader))

        val_loss, val_acc = 0.0, 0.0
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, targets = data
                inputs, targets = inputs.float().to(device), targets.float().unsqueeze(1).to(device)
        
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                predict = torch.zeros(inputs.size(0), 1)
                for i in range(inputs.size(0)):
                    if outputs[i, 0] > 0.5:
                        predict[i, 0] = 1.0

                correct = predict.cpu().eq(targets.cpu()).sum()
                acc = float(correct) / inputs.size(0)

                val_loss += loss.item()
                val_acc += acc

        VAL_LOSS.append(val_loss / len(val_loader))
        VAL_ACC.append(val_acc / len(val_loader))

        print("\n\n\nEpoch {} is done, taking {} seconds".format(epoch, time.time()-start))
        print("Train loss = {:.5f}, Train acc = {:.5f}".format(TRAIN_LOSS[-1], TRAIN_ACC[-1]))
        print("Val loss = {:.5f}, Val acc = {:.5f}\n\n".format(VAL_LOSS[-1], VAL_ACC[-1]))

        torch.save(model.state_dict(), "./ckpt/epoch{}-train_acc{:.5f}-val_acc{:.5f}.ckpt".format(epoch, TRAIN_ACC[-1], VAL_ACC[-1]))
    
    """ plt.figure(1)
    plt.plot([i+1 for i in range(EPOCH)], TRAIN_LOSS, color="r", label="train loss")
    plt.plot([i+1 for i in range(EPOCH)], VAL_LOSS, color="b", label="valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend(loc='upper right')
    plt.savefig("./ckpt/loss.png")

    plt.figure(2)
    plt.plot([i+1 for i in range(EPOCH)], TRAIN_ACC, color="r", label="train acc")
    plt.plot([i+1 for i in range(EPOCH)], VAL_ACC, color="b", label="valid acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend(loc='upper right')
    plt.savefig("./ckpt/acc.png") """