import torch
from torch import nn
from torchvision import models

class VGGLoss_ESRGAN(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19_model = models.vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])
        self.criterion = nn.L1Loss()

    def forward(self, real, fake):
        return self.criterion(self.vgg19_54(real), self.vgg19_54(fake))
