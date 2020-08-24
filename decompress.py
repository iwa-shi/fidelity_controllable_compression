import cv2
from collections import OrderedDict
from glob import glob
from itertools import product
import numpy as np
import os
from tqdm import tqdm
from scipy.special import erf

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import CompModel
from opt import opt_test
import arithmetic_coding as ac

MAX_N = 65536
TINY = 1e-10

def img_torch2np(img):
    img_np = img[0].cpu().numpy().transpose(1, 2, 0)
    img_np = (((img_np + 1.) / 2) * 255).astype(np.uint8)
    return img_np

def load_model(args):
    args.device = 'cpu'
    comp_model = CompModel(args)
    # use network interpolation
    if args.use_net_interp:
        state_dict1 = torch.load(args.model_path, map_location='cpu')['comp_model']
        state_dict2 = torch.load(args.model_path2, map_location='cpu')['comp_model']
        state_dict_interp = OrderedDict()
        for k in state_dict1:
            if 'decoder' in k:
                state_dict_interp[k] = args.interp_alpha * state_dict1[k] + (1 - args.interp_alpha) * state_dict2[k]
            else:
                state_dict_interp[k] = state_dict1[k]
        comp_model.load_state_dict(state_dict_interp)

    else:
        state_dict = torch.load(args.model_path, map_location='cpu')
        comp_model.load_state_dict(state_dict['comp_model'])
    comp_model.eval()
    return comp_model

def decompress(args):
    comp_model = load_model(args)
    
    if os.path.isdir(args.binary_path):
        pathes = glob(os.path.join(args.binary_path, '*'))
    else:
        pathes = [args.binary_path]

    for path in pathes:
        fileobj = open(path, mode='rb')
        buf = fileobj.read(4)
        arr = np.frombuffer(buf, dtype=np.uint16)
        W, H = int(arr[0]), int(arr[1])
        buf = fileobj.read(2)
        arr = np.frombuffer(buf, dtype=np.uint8)
        pad_w, pad_h = int(arr[0]), int(arr[1])
        buf = fileobj.read(1)
        arr = np.frombuffer(buf, dtype=np.uint8)
        min_val = int(arr[0])

        bitin = ac.BitInputStream(fileobj)
        dec = ac.ArithmeticDecoder(bitin)
        y_hat = torch.zeros(1, args.bottleneck, H // 16, W // 16,  dtype=torch.float32)    
        _, yC, yH, yW = y_hat.size()

        pad = nn.ZeroPad2d((2, 2, 2, 0))
        y_hat_pad = pad(y_hat)

        print('========================================================================')
        print('image', os.path.basename(path))

        with torch.no_grad():
            with tqdm(product(range(yH), range(yW)), 
                        ncols=80, total=yH*yW, desc='encode y') as qbar:
                samples = np.arange(0, min_val*2+1).reshape(-1, 1)
                for h, w in qbar:
                    p = comp_model.contextmodel.parameter_estimate(y_hat_pad[:, :, h:h+3, w:w+5])
                    p = p.numpy()
                    p = np.reshape(p, (1, args.gmm_K, args.bottleneck*3, 3, 5))
                    y_mu = p[:, :, :args.bottleneck, 2, 2] + min_val
                    y_std = np.abs(p[:, :, args.bottleneck:2*args.bottleneck, 2, 2])
                    y_w = p[:, :, 2*args.bottleneck:, 2, 2]
                    y_w = np.exp(y_w) / np.sum(np.exp(y_w), axis=1) #softmax

                    for ch in range(yC):
                        weight = y_w[:, :, ch]
                        mean = y_mu[:, :, ch]
                        std = y_std[:, :, ch]

                        high = weight * 0.5 * (1 + erf((samples + 0.5 - mean) / ((std + TINY) * 2 ** 0.5)))
                        low = weight * 0.5 * (1 + erf((samples - 0.5 - mean) / ((std + TINY) * 2 ** 0.5)))
                        pmf = np.sum(high - low, axis=1)
                        pmf_clip = np.clip(pmf, 1.0/MAX_N, 1.0)
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * MAX_N).astype(np.uint32)

                        freq = ac.SimpleFrequencyTable(pmf_clip)
                        symbol = dec.read(freq)
                        y_hat_pad[0, ch, h+2, w+2] = symbol - min_val
                        
            y_hat = y_hat_pad[:, :, 2:, 2:yW+2]
            fake_images = comp_model.decoder(y_hat)
            fake_images = fake_images[:, :, :H-pad_h, :W-pad_w]
            
            fakepath = "outputs/reconstruction/{}.jpg".format(os.path.basename(path).split('.')[0])
            cv2.imwrite(fakepath, img_torch2np(fake_images)[..., ::-1])

        print('========================================================================\n')


if __name__ == "__main__":
    args = opt_test()
    decompress(args)