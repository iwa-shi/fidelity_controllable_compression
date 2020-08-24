import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import Encoder, Decoder, Quantizer, ContextModel


class CompModel(nn.Module):
    def __init__(self, args, is_test=False):
        super().__init__()
        self.encoder = Encoder(out_channel=args.bottleneck, ch=args.main_channel, device=args.device)
        self.decoder = Decoder(in_channel=args.bottleneck, ch=args.main_channel, device=args.device)
        self.quantizer = Quantizer()
        self.contextmodel = ContextModel(device=args.device, bottleneck=args.bottleneck, gmm_K=args.gmm_K, ch=args.main_channel)
        
    def forward(self, x, mask=None):
        N, _, H, W = x.size()
        y = self.encoder(x)
        y_noise, _ = self.quantizer(y)
        out_img = self.decoder(y_noise)
        bitcost = self.contextmodel(y_noise)
        bpp = bitcost / N / H / W
        return out_img, bpp
    

    def train_only_decoder(self, x, mask=None):
        with torch.no_grad():
            y = self.encoder(x)
        out_img = self.decoder(torch.round(y))
        return out_img

    def test(self, x, mask=None):
        N, _, H, W = x.size()
        y = self.encoder(x)
        _, y_hard = self.quantizer(y)
        out_img = self.decoder(y_hard)
        bitcost = self.contextmodel(y_hard)
        bpp = bitcost / N / H / W
        return out_img, bpp

    def compress(self, x):
        y = self.encoder(x)
        _, y_hard = self.quantizer(y)
        param = self.contextmodel.parameter_estimate(y_hard)
        return y_hard, param
