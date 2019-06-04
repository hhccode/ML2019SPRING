import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConv(nn.Module):
    def __init__(self, channels, s):
        super(DepthwiseConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                kernel_size=3, stride=s, padding=1, bias=False, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU6(True)
        )

    def forward(self, x):
        return self.conv(x)

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(PointwiseConv, self).__init__()
        
        self.activation = activation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.activation:
            return F.relu6(self.conv(x), True)
        return self.conv(x)

class MobileNetV1(nn.Module):
    def __init__(self, channels=32):
        super(MobileNetV1, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.conv_layer = nn.Sequential(
            DepthwiseConv(channels, 1),
            PointwiseConv(channels, channels*2, activation=True),
            DepthwiseConv(channels*2, 1),
            PointwiseConv(channels*2, channels*2, activation=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),


            DepthwiseConv(channels*2, 1),
            PointwiseConv(channels*2, channels*4, activation=True),
            DepthwiseConv(channels*4, 1),
            PointwiseConv(channels*4, channels*4, activation=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            DepthwiseConv(channels*4, 1),
            PointwiseConv(channels*4, channels*4, activation=True),
            DepthwiseConv(channels*4, 1),
            PointwiseConv(channels*4, channels*8, activation=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.AdaptiveAvgPool2d(1)
        )

        self.output_layer = nn.Linear(channels*8, 7)


    def forward(self, x):
        x = self.input_layer(x)
        x = self.conv_layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.output_layer(x)

        return x

class CNN(nn.Module):
    def __init__(self, channels=32):
        super(CNN, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels*2, out_channels=channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
       
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=channels*2, out_channels=channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels*4, out_channels=channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.35)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=channels*4, out_channels=channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels*4, out_channels=channels*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels*8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),

            nn.AdaptiveAvgPool2d(1)
        )
        
        self.block5 = nn.Linear(channels*8, 7)
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.reshape(x.size(0), -1)
        x = self.block5(x)
     
        return x