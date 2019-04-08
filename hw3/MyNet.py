import torch
import torch.nn as nn

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        
        self.block1 = nn.Sequential(
            # Input: (1, 48, 48), Output: (64, 48, 48)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # Input: (64, 48, 48), Output: (64, 48, 48)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # Input: (64, 48, 48), Output: (64, 24, 24)
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25)
        )
        
        self.block2 = nn.Sequential(
            # Input: (64, 24, 24), Output: (128, 24, 24)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            # Input: (128, 24, 24), Output: (128, 24, 24)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            # Input: (128, 24, 24), Output: (128, 12, 12)
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )
       
        self.block3 = nn.Sequential(
            # Input: (128, 12, 12), Output: (256, 12, 12)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Input: (256, 12, 12), Output: (256, 12, 12)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Input: (256, 12, 12), Output: (256, 12, 12)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # Input: (256, 12, 12), Output: (256, 6, 6)
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.35)
        )

        self.block4 = nn.Sequential(
            # Input: (256, 6, 6), Output: (512, 6, 6)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            # Input: (512, 6, 6), Output: (512, 6, 6)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            # Input: (512, 6, 6), Output: (512, 6, 6)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            # Input: (512, 6, 6), Output: (512, 3, 3)
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.4)
        )
        self.block5 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 7)
        )
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Flatten the tensor
        x = x.view(-1, self.num_flat_features(x))
   
        output = self.block5(x)
     
        return output


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ImageDNN(nn.Module):
    def __init__(self):
        super(ImageDNN, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(48 * 48 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7)
        )

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))

        output = self.block(x)

        return output
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features