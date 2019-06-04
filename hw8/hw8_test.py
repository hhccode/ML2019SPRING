import sys
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import ImageDataset
from models import MobileNetV1

def readfile(path):
    print("Reading File...")
    id_test = []
    x_test = []

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)

        x_test.append(tmp)
        id_test.append(raw_train[i][0])
        
    x_test = np.array(x_test, dtype=float) / 255.0
    id_test = np.array(id_test, dtype=int)
    x_test = torch.FloatTensor(x_test)
    id_test = torch.LongTensor(id_test)

    return x_test, id_test

def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    x_test, id_test = readfile(argv[1])

    trans = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])

    dataset = ImageDataset(x=x_test, y=id_test, transform=trans)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    
    model = MobileNetV1()
    model.load_state_dict(torch.load("./model/mobilenet.ckpt", map_location=device))
    model.float()
    model.to(device)
    model.eval()
   
    with torch.no_grad():
        with open(argv[2], "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(["id","label"])
            for i, data in enumerate(loader):
                print("\r[{}/{}]".format(i+1, len(loader)), end="")
                inputs, ids = data
                inputs = inputs.to(device)

                outputs = model(inputs)

                predict = torch.max(outputs, 1)[1]

                for j in range(predict.size(0)):
                    writer.writerow([ids[j].item(), predict[j].item()])

if __name__ == "__main__":
    main(sys.argv)