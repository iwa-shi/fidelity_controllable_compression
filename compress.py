import cv2
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

def load_img(path):
    img = cv2.imread(path).astype(np.float32)[..., ::-1]
    img = ((img / 255.) - 0.5) * 2.
    img = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0)
    _, _, h, w = img.size()
    h_, w_ = h, w
    if h % 16 != 0:
        h_ = (h // 16 + 1) * 16
    if w % 16 != 0:
        w_ = (w // 16 + 1) * 16
    img_ = torch.zeros((1, 3, h_, w_))
    img_[:, :, :h, :w] = img
    return img_, h_ - h, w_ - w
 
def load_model(args):
    args.device = 'cpu'
    comp_model = CompModel(args)
    state_dict = torch.load(args.model_path, map_location='cpu')
    comp_model.load_state_dict(state_dict['comp_model'])
    comp_model.eval()
    return comp_model

def compress(args):
    comp_model = load_model(args)

    if os.path.isdir(args.image_path):
        pathes = glob(os.path.join(args.image_path, '*'))
    else:
        pathes = [args.image_path]

    for path in pathes:
        bitpath = "outputs/binary/{}.pth".format(os.path.basename(path).split('.')[0])

        img, pad_h, pad_w = load_img(path)
        _, _, H, W = img.size()

        with torch.no_grad():
            y_hat, p = comp_model.compress(img)
        _, yC, yH, yW = y_hat.size()
        min_val = int(torch.max(torch.abs(y_hat)))

        p = p.detach().numpy()
        p = np.reshape(p, (1, args.gmm_K, args.bottleneck*3, yH, yW))
        y_mu = p[:, :, :args.bottleneck, :, :] + min_val
        y_std = np.abs(p[:, :, args.bottleneck:2*args.bottleneck, :, :])
        y_w = p[:, :, 2*args.bottleneck:, :, :]
        y_w = np.exp(y_w) / np.sum(np.exp(y_w), axis=1) #softmax
        
        # store side information
        fileobj = open(bitpath, mode='wb')
        img_size = np.array([W, H], dtype=np.uint16)
        img_size.tofile(fileobj)
        pad_size = np.array([pad_w, pad_h], dtype=np.uint8)
        pad_size.tofile(fileobj)
        min_value = np.array([min_val], dtype=np.uint8)
        min_value.tofile(fileobj)
        fileobj.close()

        print('=============================================================')
        print(os.path.basename(path))

        with open(bitpath, 'ab+') as fout:
            bit_out = ac.CountingBitOutputStream(
                bit_out=ac.BitOutputStream(fout))
            enc = ac.ArithmeticEncoder(bit_out)

            samples = np.arange(0, min_val*2+1).reshape(-1, 1)
            with tqdm(product(range(yH), range(yW)), ncols=60, total=yH*yW) as qbar:
                for h, w in qbar:
                    for ch in range(yC):
                        weight = y_w[:, :, ch, h, w]
                        mean = y_mu[:, :, ch, h, w]
                        std = y_std[:, :, ch, h, w]

                        high = weight * 0.5 * (1 + erf((samples + 0.5 - mean) / ((std + TINY) * 2 ** 0.5)))
                        low = weight * 0.5 * (1 + erf((samples - 0.5 - mean) / ((std + TINY) * 2 ** 0.5)))
                        pmf = np.sum(high - low, axis=1)
                        pmf_clip = np.clip(pmf, 1.0/MAX_N, 1.0)
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * MAX_N).astype(np.uint32)
                        
                        symbol = np.int(y_hat[0, ch, h, w].item() + min_val)
                        freq = ac.SimpleFrequencyTable(pmf_clip)
                        enc.write(freq, symbol)
            
            enc.finish()
            bit_out.close()
        
        real_bpp = os.path.getsize(bitpath) * 8        
        print('bitrate : {0:.4}bpp'.format(real_bpp / H / W))
        print('=============================================================\n')


if __name__ == "__main__":
    args = opt_test()
    compress(args)