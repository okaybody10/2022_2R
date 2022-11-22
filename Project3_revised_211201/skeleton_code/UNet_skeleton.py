import torch.nn as nn
import torch

###########################################################################
# Implement the UNet model code.
# Understand architecture of the UNet in lecture

def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 3은 kernel size
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        ########## fill in the blanks (HINT : check out the channel size in lecture)
        self.convDown1 = conv(in_channels, #blank#)
        self.convDown2 = conv(#blank#, #blank#)
        self.convDown3 = conv(#blank#, #blank#)
        self.convDown4 = conv(#blank#, #blank#)
        self.convDown5 = conv(#blank#, #blank#)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convUp4 = conv(#blank#, #blank#)
        self.convUp3 = conv(#blank#, #blank#)
        self.convUp2 = conv(#blank#, #blank#)
        self.convUp1 = conv(#blank#, #blank#)
        self.convUp_fin = nn.Conv2d(#blank#, out_channels, 1)



    def forward(self, x):
        conv1 = self.convDown1(x)
        x = self.maxpool(conv1)
        conv2 = self.convDown2(x)
        x = self.maxpool(conv2)
        conv3 = self.convDown3(x)
        x = self.maxpool(conv3)
        conv4 = self.convDown4(x)
        x = self.maxpool(conv4)
        conv5 = self.convDown5(x)
        x = self.upsample(conv5)
        #######fill in here #######
        # HINT : concatenation (Lecture slides)
        x = self.convUp4(x)
        x = self.upsample(x)
        #######fill in here ####### 
        # HINT : concatenation (Lecture slides)
        x = self.convUp3(x)
        x = self.upsample(x)
        #######fill in here ####### 
        # HINT : concatenation (Lecture slides)
        x = self.convUp2(x)
        x = self.upsample(x)
        #######fill in here ####### 
        # HINT : concatenation (Lecture slides)
        x = self.convUp1(x)
        out = self.convUp_fin(x)

        return out