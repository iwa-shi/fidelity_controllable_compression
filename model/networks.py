import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import NLAM, ResBlock, UpResBlock, MaskConv2d


class Encoder(nn.Module):
    def __init__(self, out_channel=32, ch=192, device='cuda:0'):
        super().__init__()
        self.block1 = ResBlock(3, ch, actv='lrelu', actv2='gdn', device=device, downscale=True)
        self.block2 = ResBlock(ch, ch, actv='lrelu', actv2='gdn', device=device)
        self.block3 = ResBlock(ch, ch, actv='lrelu', actv2='gdn', device=device, downscale=True)
        self.nlam1 = NLAM(ch, use_nln=False)
        self.block4 = ResBlock(ch, ch, actv='lrelu', actv2='gdn', device=device)
        self.block5 = ResBlock(ch, ch, actv='lrelu', actv2='gdn', device=device, downscale=True)
        self.block6 = ResBlock(ch, ch, actv='lrelu', actv2='gdn', device=device)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.nlam2 = NLAM(ch, use_nln=False)
        self.conv2 = nn.Conv2d(ch, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.nlam1(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv1(x)
        x = self.nlam2(x)
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channel=32, ch=192, device='cuda:0', last_conv=False, use_noise=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, ch, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.nlam1 = NLAM(ch, use_nln=False)
        self.block1 = ResBlock(ch, ch, actv='lrelu')
        self.up1 = UpResBlock(ch, ch, actv='lrelu', actv2='igdn', device=device)
        self.block2 = ResBlock(ch, ch, actv='lrelu')
        self.up2 = UpResBlock(ch, ch, actv='lrelu', actv2='igdn', device=device)
        self.nlam2 = NLAM(ch, use_nln=False)
        self.block3 = ResBlock(ch, ch, actv='lrelu')
        self.up3 = UpResBlock(ch, ch, actv='lrelu', actv2='igdn', device=device)
        self.block4 = ResBlock(ch, ch, actv='lrelu')
        self.up4 = nn.Sequential(
            nn.Conv2d(ch, 12, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.nlam1(x)
        x = self.block1(x)
        x = self.up1(x)
        x = self.block2(x)
        x = self.up2(x)
        x = self.nlam2(x)
        x = self.block3(x)
        x = self.up3(x)
        x = self.block4(x)
        x = self.up4(x)
        return x

class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        hard = torch.round(x)
        noise = x + torch.rand_like(x) - 0.5
        return noise, hard

class ContextModel(nn.Module):
    def __init__(self, device='cuda:0', bottleneck=32, gmm_K=3, ch=192):
        super().__init__()
        self.mask_conv1 = MaskConv2d(in_channel=bottleneck, out_channel=ch*2, kernel_size=5, mask_type='first', device=device)
        self.c1 = nn.Conv2d(ch*2, 640, kernel_size=1, stride=1)
        self.c2 = nn.Conv2d(640, 640, kernel_size=1, stride=1)
        self.c3 = nn.Conv2d(640, 3*gmm_K*bottleneck, kernel_size=1, stride=1)
        self.actv = nn.LeakyReLU(0.2, inplace=True)
        self.K = gmm_K
        self.bottleneck = bottleneck
    
    def forward(self, y_hat):
        p = self.parameter_estimate(y_hat)
        bc = self.bitcost(y_hat, p)
        return bc

    def parameter_estimate(self, y_hat, padding=True):
        a = self.mask_conv1(y_hat, padding=padding)
        a = self.actv(a)
        a = self.c1(a)
        a = self.actv(a)
        a = self.c2(a)
        a = self.actv(a)
        p = self.c3(a)
        return p

    def get_gmm_params(self, y_hat):
        p = self.parameter_estimate(y_hat, padding=False)
        p = p.numpy()
        p = np.reshape(p, (1, self.K, self.bottleneck*3, 1, 1))
        mu = p[:, :, :self.bottleneck, 0, 0]
        std = np.abs(p[:, :, self.bottleneck:2*self.bottleneck, 0, 0])
        w = p[:, :, 2*self.bottleneck:, 0, 0]
        w = np.exp(w) / np.sum(np.exp(w), axis=1) #softmax
        return mu, std, w

    def bitcost(self, y_hat, p):
        N, _, H, W = p.size()
        p = p.view(N, self.K, self.bottleneck*3, H, W)
        mu = p[:, :, :self.bottleneck, :, :]
        std = p[:, :, self.bottleneck:2*self.bottleneck, :, :]
        w = p[:, :, 2*self.bottleneck:, :, :]
        w = F.softmax(w, dim=1)
        
        total_diff = 1e-6
        for k in range(self.K):
            weight_k = w[:, k, :, :, :]
            mu_k = mu[:, k, :, :, :]
            std_k = torch.abs(std[:, k, :, :, :]) + 1e-6

            cml_high = 0.5 * (1 + torch.erf((y_hat + 0.5 - mu_k) * std_k.reciprocal() / np.sqrt(2)))
            cml_low = 0.5 * (1 + torch.erf((y_hat - 0.5 - mu_k) * std_k.reciprocal() / np.sqrt(2)))
            dif = cml_high - cml_low
            total_diff += dif * weight_k

        bc_y = - torch.sum(torch.log(total_diff)) / np.log(2)
        return bc_y

    