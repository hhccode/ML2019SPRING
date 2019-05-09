import sys
import csv
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from models import DNN
from datasets import BOW

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    dataset = BOW(x_path=sys.argv[1], y_path=None, dict_path=sys.argv[2], load=True)
    loader = DataLoader(dataset, batch_size=64)

    model = DNN()
    model.to(device)

    predict = np.array([]).reshape(0, 1)

    with torch.no_grad():

        model.load_state_dict(torch.load("./model/BOW+DNN/model.ckpt"))
        model.eval()

        for i, data in enumerate(loader):
            inputs = data
            inputs = inputs.float().to(device)
            
            outputs = model(inputs).cpu().numpy()
            
            predict = np.concatenate((predict, outputs))


    with open(sys.argv[3], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        id = 0

        for i in range(predict.shape[0]):
            if predict[i] > 0.5:
                writer.writerow([id, 1])
            else:
                writer.writerow([id, 0])
            id += 1


            



        