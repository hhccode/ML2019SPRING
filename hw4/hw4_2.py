import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
import matplotlib.pyplot as plt
import MyDataset

if __name__ == "__main__":
    device = torch.device("cuda")
    model = torch.load("./bestmodel.pkl?dl=1")
    
    # Take partial of the original model
    sub_model = nn.Sequential(
        *list(model.children())[0][:6],
    )
    sub_model.to(device)
    sub_model.eval()

    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2], exist_ok=True)

    EPOCH = 100
    FILTER_NUMBER = 64
    fig = plt.figure(num=1, figsize=(14,8))
    
    # Find the input which can maximum the activation of each filter in first layer
    for filter_num in range(FILTER_NUMBER):
        # Declare a white noise input
        x = torch.load("initial.pt", map_location=device)

        optimizer = Adam(params=[x], lr=1, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(EPOCH):
            optimizer.zero_grad()

            output = sub_model(x).squeeze(0)
            activation = torch.sum(output[filter_num])
            (-activation).backward()
            
            optimizer.step()

        filter_img = x.squeeze().detach().cpu().numpy()

        ax = fig.add_subplot(4, 16, filter_num+1)
        ax.imshow(filter_img, cmap="Oranges")
        plt.xticks(np.array([]))
        plt.xlabel("Filter {}".format(filter_num+1))
        plt.yticks(np.array([]))
        plt.tight_layout()


        print("Finish finding filter {}.".format(filter_num))
        
    fig.suptitle("Fig.2-1\nFilters of 2nd Conv2d in block1 (# of ascent epoch: {})".format(EPOCH))
    fig.savefig(os.path.join(sys.argv[2], "fig2_1.jpg"))
    

    data = MyDataset.ImageDataset(sys.argv[1], transform=transforms.ToTensor())[6987]

    outputs = sub_model(data[0].unsqueeze(0).to(device)).squeeze(0)

    fig = plt.figure(num=2, figsize=(14,8))

    # Given certain image input, find the output of each filter in first layer
    for filter_num in range(FILTER_NUMBER):
        output = outputs[filter_num].detach().cpu().numpy()
        
        ax = fig.add_subplot(4, 16, filter_num+1)
        ax.imshow(output, cmap="Oranges")
        plt.xticks(np.array([]))
        plt.xlabel("Filter {}".format(filter_num+1))
        plt.yticks(np.array([]))
        plt.tight_layout()

        print("Finish finding the output of filter {}.".format(filter_num))

    fig.suptitle("Fig. 2-2\nOutputs of 2nd Conv2d in block1 (Given image 6987)")
    fig.savefig(os.path.join(sys.argv[2], "fig2_2.jpg"))