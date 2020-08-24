import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pytorch_gdn import GDN

class ResBlock(nn.Module):
    def __init__(self, in_channel=192, out_channel=192, actv='relu', actv2=None, downscale=False, kernel_size=3, device='cuda:0'):
        super().__init__()
        stride = 2 if downscale else 1
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=1, stride=1)
        if actv == 'relu':
            self.actv1 = nn.ReLU(inplace=True)
        elif actv == 'lrelu':
            self.actv1 = nn.LeakyReLU(0.2, inplace=True)

        if actv2 is None:
            self.actv2 = None
        elif actv2 == 'gdn':
            self.actv2 = GDN(out_channel, device)
        elif actv2 == 'igdn':
            self.actv2 = GDN(out_channel, device, inverse=True)
        
        self.downscale = downscale
        if self.downscale:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2)

    def forward(self, x):
        shortcut = x
        if self.downscale:
            shortcut = self.shortcut(shortcut)
        x = self.conv1(x)
        x = self.actv1(x)
        x = self.conv2(x)
        if self.actv2 is not None:
            x = self.actv2(x)
        x = x + shortcut
        return x


class NLAM(nn.Module):
    def __init__(self, channel=192, down_scale=1, use_nln=True):
        super().__init__()
        self.blocks_mask = nn.Sequential(
            ResBlock(channel, channel),
            ResBlock(channel, channel),
            ResBlock(channel, channel)
        )
        self.conv_mask = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.blocks_main = nn.Sequential(
            ResBlock(channel, channel),
            ResBlock(channel, channel),
            ResBlock(channel, channel)
        )

    def forward(self, x):
        mask = x
        mask = self.blocks_mask(mask)
        mask = self.conv_mask(mask)
        mask = self.sigmoid(mask)
        main = self.blocks_main(x)
        out = main * mask + x
        return out


class UpResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, actv='relu', actv2=None, device='cuda:0', up_type='pixelshuffle'):
        super().__init__()
        pad = (kernel_size - 1) // 2
        if actv == 'relu':
            actv1 = nn.ReLU(inplace=True)
        elif actv == 'lrelu':
            actv1 = nn.LeakyReLU(0.2, inplace=True)
        
        main_layers = [
            nn.Conv2d(in_channel, out_channel*4, kernel_size=kernel_size, padding=pad),
            actv1,
            nn.PixelShuffle(2),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=pad),
        ]

        if actv2 is not None:
            if actv2 == 'igdn':
                act2 = GDN(out_channel, device, inverse=True)
            elif actv2 == 'gdn':
                act2 = GDN(out_channel, device)
            main_layers += [act2]

        self.c1 = nn.Sequential(*main_layers)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.c1(x)
        return x + shortcut


class MaskConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, mask_type=None, device='cuda:0'):
        super().__init__()
        pad = (kernel_size // 2, kernel_size // 2, kernel_size // 2, 0)
        self.padding = nn.ZeroPad2d(pad)
        kernel_shape = (kernel_size // 2 + 1, kernel_size)
        self.mask = self.get_mask(kernel_size, mask_type).to(device)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_shape, stride, padding=0, bias=True)

    
    def get_mask(self, k, mask_type='first'):
        c = 0 if mask_type == 'first' else 1
        mask = np.ones((k // 2 + 1, k), dtype=np.float32)
        mask[k // 2, k // 2 + c:] = 0
        mask[k // 2 + 1:, :] = 0
        mask = torch.from_numpy(mask).unsqueeze(0)
        return mask

    def forward(self, x):
        x = self.padding(x)
        self.conv.weight.data = self.conv.weight.data * self.mask
        return self.conv(x)