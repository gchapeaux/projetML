
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual (nn.Module):
    def __init__(self, padding = True):
        super(Residual, self).__init__()
        self.couche1 = nn.Conv2d(128, 128, kernel_size=3, padding= 1 if padding else 0)
        self.batchNorm1 = nn.BatchNorm2d(128)
        self.couche2 = nn.Conv2d(128, 128, kernel_size=3, padding= 1 if padding else 0)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.ReLU = F.relu
        self.padding = padding
        
    def forward(self, x):
        y = self.ReLU(self.batchNorm1(self.couche1(x)))
        y = self.batchNorm2(self.couche2(y))
        if self.padding :
            return y + x
        else:
            return self.ReLU(y)

def STanh(x):
    return 255/2*(1+nn.Tanh()(x))

class ImgTNet (nn.Module):
    def __init__(self):
        super(ImgTNet, self).__init__()
        self.reflectionPadding = torch.nn.ReflectionPad2d(40)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.residual1 = Residual(False)
        self.residual2 = Residual(False)
        self.residual3 = Residual(False)
        self.residual4 = Residual(False)
        self.residual5 = Residual(False)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding = 1, output_padding = 1)
        self.batchNorm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding = 1, output_padding = 1)
        self.batchNorm5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1, padding = 4)
        self.batchNorm6 = nn.BatchNorm2d(3)
        self.ReLU = F.relu

    def forward(self, x):
        out = self.reflectionPadding(x)
        out = self.ReLU(self.batchNorm1(self.conv1(out)))
        out = self.ReLU(self.batchNorm2(self.conv2(out)))
        out = self.ReLU(self.batchNorm3(self.conv3(out)))
        out = self.residual1.forward(out)
        out = self.residual2.forward(out)
        out = self.residual3.forward(out)
        out = self.residual4.forward(out)
        out = self.residual5.forward(out)
        out = self.ReLU(self.batchNorm4(self.conv4(out)))
        out = self.ReLU(self.batchNorm5(self.conv5(out)))
        out = STanh(self.batchNorm6(self.conv6(out)))
        return out
        