import numpy as np
from PIL import Image
import requests
from io import BytesIO
from matplotlib import pyplot as plt
from time import time

import pickle
import urllib.request
import json
import os
import copy
from tqdm import tqdm, trange

import os
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


from models import Residual
from models import ImgTNet

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

unloader = transforms.ToPILImage()  # reconvert into PIL image

def save_model(reseau, classe= None, filename = "mini_trained3.pickl"):
    if classe is None:
        pickle.dump(reseau, open(filename,'wb'))
    else:
        model_clone = classe()
        model_clone.load_state_dict(copy.deepcopy(reseau.state_dict()))
        pickle.dump(model_clone, open(filename,'wb'))

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
        return (1 /(c * h * w)) * torch.pow((x - self.target), 2).sum()

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.compute_loss = nn.MSELoss()
        self.target = target.detach()

    def forward(self, x):
        channel = x.shape[1]
        return ( 1 / channel ** 2) * self.compute_loss(GramMatrix()(x), GramMatrix()(self.target))

class OutputLayer(nn.Module):
    def __init__(self):
        super(OutputLayer, self).__init__()
        
    def forward(self, x):
        self.output = x
        return x
    
class VGG_net(nn.Module):
    def __init__(self, children_list, content_layers, style_layers,               
                 normalization_mean = cnn_normalization_mean, 
                 normalization_std = cnn_normalization_std,
                StyleLoss = StyleLoss,
                ContentLoss = ContentLoss):
        super().__init__()
        i = 0
        imax = max([int(name.split('_')[-1]) for name in (content_layers+style_layers)])
        self.vgg = nn.Sequential(Normalization(normalization_mean, normalization_std).to(device))
        self.content_losses = []
        self.style_losses = []
        for layer in children_list:
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
                content_loss = OutputLayer()
                self.vgg.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                style_loss = OutputLayer()
                self.vgg.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)
                
            
    def forward(self, x):
        self.vgg(x)
        return [style.output for  style in self.style_losses], [content.output for  content in self.content_losses]
    
    
def checkimages():
    images = os.listdir('dataset')
    for filename in images:
        if not image_loader('dataset/'+filename).shape[1]==3:
            print(filename)
def getimages():
    images = os.listdir('dataset')
    random.shuffle(images)
    #images = images[:len(images)//10]
    return torch.cat([image_loader('dataset/'+filename) for filename in images if image_loader('dataset/'+filename).shape [1] == 3], 0)

def get_cnn_vgg():
    return models.vgg19(pretrained=True).features.to(device).eval()

def train_model(style_img, output_model, images,                            
                       normalization_mean = cnn_normalization_mean, 
                       normalization_std =cnn_normalization_std,
                       num_steps=1,
                       style_weight=1000000, 
                       content_weight=10, step = 50, save = False, show = False,
                       content_layers=content_layers_default,
                       style_layers=style_layers_default, repeat = 1, lr = 10**-8,momentum=0.95, fraction = 1, batch_size = 2):
    """Run the style transfer."""
    optimizer =  optim.SGD(output_model.parameters(), lr=lr, momentum=momentum)
    
    print('Loading image dataset')
    
    #Lol non en fait
    
    print('Building the loss model..')
    global loss_model
    vgg_model = VGG_net(get_cnn_vgg().children(), content_layers, style_layers,                      
                 normalization_mean = normalization_mean, 
                 normalization_std = normalization_std)
    global style_outputs
    global style_pred
    global content_outputs
    global content_pred
    style_outputs = vgg_model(style_img)[0]
    print('Optimizing..')
    global iteration
    permutation = torch.randperm(images.size()[0])
    for epoch in range(num_steps):
        totloss = 0
        N = 0
        for i in range(0, images.size()[0]//fraction, batch_size):
            for iteration in range(repeat):
                batch = images[permutation[i:i+batch_size]]
                optimizer.zero_grad()

                result_images = output_model(batch)
                content_outputs = vgg_model(batch)[1]
                
                style_pred, content_pred = vgg_model(result_images)
                
                style_loss = 0
                content_loss = 0
                for i in range(len(style_pred)):
                    for j in range(style_pred[i].shape[0]):
                        style_loss += StyleLoss(style_outputs[i])(style_pred[i][j:j+1])
                for i in range(len(content_pred)):
                    content_loss += ContentLoss(content_outputs[i])(content_pred[i])
                
                loss = style_loss * style_weight + content_loss * content_weight

                loss.backward()

                optimizer.step()
                totloss += loss.item()
                N+=batch.shape[0]
        print("average loss", totloss/N)

imsize = 256
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


if __name__ == "__main__":
    filename = "mini_trained6.pickl"
    out_filename = "mini_trained7.pickl"
    style_path = "images/wave.jpeg"

    style_image = image_loader(style_path)

    print("loading images")
    images = getimages()

    print("reset du réseau") 
    reseau = pickle.load(open(filename,'rb')).to(device)

    d = time()
    train_model(style_image, reseau, images, num_steps= 20, repeat = 1, save = True, lr = 10**-8, batch_size = 2)
    print("temps total", time()-d)


    save_model(reseau, filename=out_filename)