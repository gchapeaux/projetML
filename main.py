import numpy as np
from PIL import Image
import requests
from io import BytesIO
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import math
import copy
cuda = True
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
        # channel = x.shape[3]  # 1 isn't it ??
        channel = x.shape[1]
        self.loss = ( 1 / channel ** 2) * self.compute_loss(GramMatrix()(x), GramMatrix()(self.target))
        return x

    

def get_style_model_and_losses(cnn, style_img, content_img,                               
                               normalization_mean = cnn_normalization_mean, 
                               normalization_std =cnn_normalization_std,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
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

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, style_img, content_img, input_img,                               
                       normalization_mean = cnn_normalization_mean, 
                       normalization_std =cnn_normalization_std,
                       num_steps=300,
                       style_weight=1000000, 
                       content_weight=10, step = 50, save = False, show = True):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img,
                                                                     normalization_mean, normalization_std)
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
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            iteration += 1
            if iteration % step == 0:
                print("iteration {}:".format(iteration))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                if show:
                    imshow(input_img)
                if save:
                    tensorToPil(input_img).save(f"output/output_{iteration // step}.jpg")
                print()
                
            return style_score + content_score
        
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

imsize = 400 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])

style_image = image_loader(style_path)
content_image = image_loader(content_path)
input_image = content_image.clone()
cnn = models.vgg19(pretrained=True).features.to(device).eval()
output = run_style_transfer(cnn, style_image, content_image, input_image, num_steps= 500, step = 50, save = True)