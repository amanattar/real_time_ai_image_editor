import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import io
import cv2
# from model import Generator  # Your custom RRDB Generator model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image

import torch
import torch.nn as nn

class RRDB(nn.Module):
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return x + out  # Residual connection

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_rrdb=23):
        super(Generator, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(64) for _ in range(num_rrdb)])
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        initial_feature = self.initial_conv(x)
        out = self.rrdb_blocks(initial_feature)
        out = self.final_conv(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, stride=1, padding=1)
        )

    def forward(self, img):
        return self.model(img)
    
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, sr, hr):
        return F.mse_loss(sr, hr)

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg_model.features[:36]  # Use pre-trained VGG features
        self.vgg.eval()

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return F.mse_loss(sr_features, hr_features)
    

def super_resolve(file):
    # Load the generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator()
    generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
    generator.to(device)
    generator.eval()

    # Load the image
    image = Image.open(file).convert("RGB")
    height, width = image.size
    transform = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Generate super-resolved image
    with torch.no_grad():
        sr_image = generator(input_tensor)

    # Convert to PIL Image for Streamlit
    buffer = io.BytesIO()
    save_image(sr_image, buffer, format="PNG")
    buffer.seek(0)
    return buffer
