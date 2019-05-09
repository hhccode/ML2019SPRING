import sys
import csv
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from models import RNN
from datasets import DcardDataset

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = RNN(embedding_dim=128, hidden_dim=256)
    model.to(device)

    predict = np.array([]).reshape(0, 1)

    with torch.no_grad():
        dataset = DcardDataset(x_path=sys.argv[1], y_path=None, dict_path=sys.argv[2], w2v_path="./model/RNN/word2vec_mincount3.model", w2v_pretrain=True)
        loader = DataLoader(dataset, batch_size=64)

        model.load_state_dict(torch.load("./model/RNN/mincount3_model.ckpt", map_location=device))
        model.eval()

        for i, data in enumerate(loader):
            inputs = data
            inputs = inputs.to(device)
        
            outputs = model(inputs).cpu().numpy()
            
            predict = np.concatenate((predict, outputs))

        dataset = DcardDataset(x_path=sys.argv[1], y_path=None, dict_path=sys.argv[2], w2v_path="./model/RNN/word2vec_mincount5.model", w2v_pretrain=True)
        loader = DataLoader(dataset, batch_size=64)
        
        model.load_state_dict(torch.load("./model/RNN/mincount5_model.ckpt", map_location=device))
        model.eval()
        
        for i, data in enumerate(loader):
            inputs = data
            inputs = inputs.to(device)
        
            outputs = model(inputs).cpu().numpy()
            
            predict[i*64:(i+1)*64, :] = predict[i*64:(i+1)*64 :] + outputs

    with open(sys.argv[3], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label'])
        id = 0

        for i in range(predict.shape[0]):
            if predict[i] > 1.0:
                writer.writerow([id, 1])
            else:
                writer.writerow([id, 0])
            id += 1


            



        