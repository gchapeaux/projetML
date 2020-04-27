from train import imshow, image_loader, tensorToPil
import torch
import pickle
from models import ImgTNet, Residual

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
filename = "mini_trained6.pickl"
image_file = "images/danse.jpg"
output_file = "output/output_final.jpg"

reseau = pickle.load(open(filename,'rb')).to(device)

tensorToPil(reseau(image_loader(image_file))).save(output_file)
