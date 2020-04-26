import numpy as np
from PIL import Image
import requests
from io import BytesIO
from matplotlib import pyplot as plt

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import math
import copy
cuda = False
device = torch.device("cuda" if cuda else "cpu")

unloader = transforms.ToPILImage()  # reconvert into PIL image

def tensorToPil(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    return  unloader(image)
def imshow(tensor, title=None):
    plt.imshow(tensorToPil(tensor))
    plt.pause(0.001)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class GramMatrix(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        features = x.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return 1/ (h * w) * G



class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        b, c, h, w = x.shape
        self.loss = (1 /(c * h * w)) * torch.pow((x - self.target), 2).sum()
        return x

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.compute_loss = nn.MSELoss()
        self.target = target.detach()

    def forward(self, x):
        channel = x.shape[1]
        self.loss = ( 1 / channel ** 2) * self.compute_loss(GramMatrix()(x), GramMatrix()(self.target))
        return x

class LossNet (nn.Module):
    def __init__(self, children_list, content_layers, style_layers, style_img, content_img,
                 style_weight=1000000, content_weight=10,
                 normalization_mean = cnn_normalization_mean,
                 normalization_std = cnn_normalization_std,
                StyleLoss = StyleLoss,
                ContentLoss = ContentLoss):
        super().__init__()
        self.style_weight=style_weight
        self.content_weight=content_weight
        i = 0
        imax = max([int(name.split('_')[-1]) for name in (content_layers+style_layers)])
        self.vgg = nn.Sequential(Normalization(normalization_mean, normalization_std).to(device))
        self.content_losses = []
        self.style_losses = []
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)

            if i > imax:
                break

            self.vgg.add_module(name, layer)


            if name in content_layers:
                # add content loss:
                target = self.vgg(content_img).detach()
                content_loss = ContentLoss(target)
                self.vgg.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = self.vgg(style_img).detach()
                style_loss = StyleLoss(target_feature)
                self.vgg.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)


    def forward(self, x):
        self.vgg(x)

        content_loss = 0
        style_loss = 0
        for content in self.content_losses:
            content_loss += content.loss

        for style in self.style_losses:
            style_loss += style.loss


        self.style_score = style_loss * self.style_weight
        self.content_score = content_loss * self.content_weight
        loss = self.style_score + self.content_score
        return loss

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
        self.STanh = lambda x : 255/2*(1+nn.Tanh()(x))

    def forward(self, x):
        verbose = False
        out = self.reflectionPadding(x)
        if verbose:
            print(out.shape)
        out = self.ReLU(self.batchNorm1(self.conv1(out)))
        if verbose:
            print(out.shape)
        out = self.ReLU(self.batchNorm2(self.conv2(out)))
        if verbose:
            print(out.shape)
        out = self.ReLU(self.batchNorm3(self.conv3(out)))
        if verbose:
            print(out.shape)
        out = self.residual1.forward(out)
        if verbose:
            print(out.shape)
        out = self.residual2.forward(out)
        if verbose:
            print(out.shape)
        out = self.residual3.forward(out)
        if verbose:
            print(out.shape)
        out = self.residual4.forward(out)
        if verbose:
            print(out.shape)
        out = self.residual5.forward(out)
        if verbose:
            print(out.shape)
        out = self.ReLU(self.batchNorm4(self.conv4(out)))
        if verbose:
            print(out.shape)
        out = self.ReLU(self.batchNorm5(self.conv5(out)))
        if verbose:
            print(out.shape)
        out = self.STanh(self.batchNorm6(self.conv6(out)))
        if verbose:
            print(out.shape)
        return out

class ImgTNet_v2 (nn.Module):
    def __init__(self):
        super(ImgTNet_v2, self).__init__()
        self.reflectionPadding = torch.nn.ReflectionPad2d(40)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.instanceNorm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.instanceNorm2 = nn.InstanceNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.instanceNorm3 = nn.InstanceNorm2d(128)
        self.residual1 = Residual(False)
        self.residual2 = Residual(False)
        self.residual3 = Residual(False)
        self.residual4 = Residual(False)
        self.residual5 = Residual(False)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding = 1, output_padding = 1)
        self.instanceNorm4 = nn.InstanceNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding = 1, output_padding = 1)
        self.instanceNorm5 = nn.InstanceNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1, padding = 4)
        self.instanceNorm6 = nn.InstanceNorm2d(3)
        self.ReLU = F.relu
        self.STanh = lambda x : 255/2*(1+nn.Tanh()(x))

    def forward(self, x):
        verbose = False
        out = self.reflectionPadding(x)
        if verbose:
            print(out.shape)
        out = self.ReLU(self.instanceNorm1(self.conv1(out)))
        if verbose:
            print(out.shape)
        out = self.ReLU(self.instanceNorm2(self.conv2(out)))
        if verbose:
            print(out.shape)
        out = self.ReLU(self.instanceNorm3(self.conv3(out)))
        if verbose:
            print(out.shape)
        out = self.residual1.forward(out)
        if verbose:
            print(out.shape)
        out = self.residual2.forward(out)
        if verbose:
            print(out.shape)
        out = self.residual3.forward(out)
        if verbose:
            print(out.shape)
        out = self.residual4.forward(out)
        if verbose:
            print(out.shape)
        out = self.residual5.forward(out)
        if verbose:
            print(out.shape)
        out = self.ReLU(self.instanceNorm4(self.conv4(out)))
        if verbose:
            print(out.shape)
        out = self.ReLU(self.instanceNorm5(self.conv5(out)))
        if verbose:
            print(out.shape)
        out = self.STanh(self.instanceNorm6(self.conv6(out)))
        if verbose:
            print(out.shape)
        return out


def run_style_transfer(cnn,network, style_img, content_img, input_img,
                       normalization_mean = cnn_normalization_mean,
                       normalization_std =cnn_normalization_std,
                       num_steps=300,
                       style_weight=1000000,
                       content_weight=10, step = 50, save = False, show = False,
                       content_layers=content_layers_default,
                       style_layers=style_layers_default):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model = network(cnn.children(), content_layers, style_layers, style_img, content_img,
                 style_weight = style_weight, content_weight = content_weight,
                 normalization_mean = normalization_mean,
                 normalization_std = normalization_std)

    optimizer =  optim.LBFGS([input_img.requires_grad_()])
    if save :
        tensorToPil(content_img).save("output/content_input.jpg")
        tensorToPil(style_img).save("output/style_input.jpg")

    print('Optimizing..')
    global iteration
    iteration = 0
    index = 1
    while iteration <= num_steps:

        def closure():
            global iteration
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            loss = model(input_img)

            loss.backward()
            iteration += 1
            if iteration % step == 0:
                print("iteration {}:".format(iteration))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    model.style_score.item(), model.content_score.item()))
                if show:
                    imshow(input_img)
                if save:
                    tensorToPil(input_img).save("output/output_"+str(iteration // step)+".jpg")
                print()

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    if save:
        tensorToPil(input_img).save("output/output_final.jpg")


    return input_img


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

content_path   = "images/danse.jpg"
style_path = "images/wave.jpeg"

#TODO : Uncomment
imsize = 256 #Network test
#imsize = 400 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])

style_image = image_loader(style_path)
content_image = image_loader(content_path)
input_image = content_image.clone()
#TODO : Uncomment
#cnn = models.vgg19(pretrained=True).features.to(device).eval()
#output = run_style_transfer(cnn,LossNet, style_image, content_image, input_image, num_steps= 150, step = 50, save = True)

def Output(network):
  if cuda :
    reseau = network().cuda()
  else :
    reseau = network().cpu()

  image = reseau(input_image)
  print(image.shape)
  tensorToPil(image)
  imshow(network().forward(content_image))

#Test method Johnson
Output(ImgTNet)

#Better version
Output(ImgTNet_v2)




