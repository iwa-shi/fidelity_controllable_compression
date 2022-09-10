import argparse
import json
import os
import sys
import tempfile
from collections import OrderedDict
from itertools import product
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict
from scipy.special import erf
from tqdm import tqdm

from model.model import CompModel
from rangecoder import RangeDecoder, RangeEncoder

MAX_N = 65536
TINY = 1e-10
HEADER_SIZE = 5

#####################################################################
##                        Codec Utils
#####################################################################
def encode_header(H, W, offset):
    info_list = [
            np.array([H, W], dtype=np.uint16),
            np.array(offset, dtype=np.uint8),
    ]
    with tempfile.TemporaryFile() as f:
        for info in info_list:
            f.write(info.tobytes())
        f.seek(0)
        header_str = f.read()
    return header_str

def decode_header(header_bytes):
    img_size_buffer = header_bytes[:4]
    img_size = np.frombuffer(img_size_buffer, dtype=np.uint16)
    H, W = int(img_size[0]), int(img_size[1])
    offset_buffer = header_bytes[4:5]
    offset = np.frombuffer(offset_buffer, dtype=np.uint8)
    offset = int(offset)
    return H, W, offset

def save_byte_strings(save_path: str, string_list: List) -> None:
    with open(save_path, 'wb') as f:
        for string in string_list:
            f.write(string)

def load_byte_strings(load_path: str) -> List[bytes]:
    out_list = []
    with open(load_path, 'rb') as f:
        header = f.read(HEADER_SIZE)
        out_list.append(header)
        out_list.append(f.read())
    return out_list


#####################################################################
##                         Image utils
#####################################################################

def load_img(img_path):
    img = cv2.imread(img_path).astype(np.float32)[..., ::-1]
    img = ((img / 255.) - 0.5) * 2.
    img = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0)
    return img

def pad_img(img, stride=16):
    _, _, h, w = img.size()
    h_ = int(np.ceil(h / stride) * stride)
    w_ = int(np.ceil(w / stride) * stride)
    img_ = torch.zeros((1, 3, h_, w_))
    img_[:, :, :h, :w] = img
    return img_

def img_torch2np(img):
    img_np = img[0].cpu().numpy().transpose(1, 2, 0)
    img_np = (((img_np + 1.) / 2) * 255).astype(np.uint8)
    return img_np[..., ::-1]


def load_model(opt):
    opt.device = 'cpu'
    comp_model = CompModel(opt)

    if opt.get('interp_model', None):
        state_dict1 = torch.load(opt.model, map_location='cpu')['comp_model']
        state_dict2 = torch.load(opt.interp_model, map_location='cpu')['comp_model']
        state_dict = OrderedDict()
        for k in state_dict1:
            if 'decoder' in k:
                state_dict[k] = opt.interp_alpha * state_dict1[k] + (1 - opt.interp_alpha) * state_dict2[k]
            else:
                state_dict[k] = state_dict1[k]
    else:
        state_dict = torch.load(opt.model, map_location='cpu')['comp_model']
    
    comp_model.load_state_dict(state_dict)
    comp_model.eval()
    return comp_model

def get_gmm_qcdf(samples, mu, std, weight):
    cdf = weight * 0.5 * (1 + erf((samples - mu + 0.5) / ((std + TINY) * 2 ** 0.5)))
    cdf = np.sum(cdf, axis=1)
    qcdf = (cdf * MAX_N).astype(np.int32)
    return qcdf

def load_opt(config_path):
    with open(config_path, 'r') as f:
        opt = json.load(f)
    opt = Dict(opt)
    return opt

@torch.no_grad()
def compress_img(comp_model, img, bin_path):
    _, _, H, W = img.size()
    img_pad = pad_img(img, stride=16)

    y_hat = comp_model.compress(img_pad)
    offset = int(torch.max(torch.abs(y_hat)))
    header_str = encode_header(H, W, offset)
    _, yC, yH, yW = y_hat.size()

    y_hat_pad = F.pad(y_hat, (2, 2, 2, 0), "constant", 0)

    samples = np.arange(0, offset*2+1).reshape(-1, 1, 1) - offset
    
    with RangeEncoder() as enc:
        with tqdm(product(range(yH), range(yW)), ncols=80, total=yH*yW) as qbar:
            for h, w in qbar:
                y_mu, y_std, y_w = comp_model.contextmodel.get_gmm_params(y_hat_pad[:, :, h:h+3, w:w+5])
                qcdf = get_gmm_qcdf(samples, y_mu, y_std, y_w)

                for ch in range(yC):
                    symbol = np.int(y_hat[0, ch, h, w].item() + offset)
                    enc.encode([symbol], qcdf[:, ch], is_normalized=False)

        y_str = enc.get_byte_string()

    string_list = [header_str, y_str]
    save_byte_strings(bin_path, string_list)
    num_bit = os.path.getsize(bin_path) * 8
    bpp = num_bit / H / W
    return num_bit, bpp

@torch.no_grad()
def decompress_img(comp_model, bin_path, bottleneck):
    str_list = load_byte_strings(bin_path)
    header_str, y_str = str_list[0], str_list[1]
    
    H, W, offset = decode_header(header_str)
    samples = np.arange(0, offset*2+1).reshape(-1, 1, 1) - offset
    y_hat = torch.zeros(1, bottleneck, int(np.ceil(H / 16)), int(np.ceil(W / 16)), dtype=torch.float)
    _, yC, yH, yW = y_hat.size()

    y_hat_pad = F.pad(y_hat, (2, 2, 2, 0), "constant", 0)

    with RangeDecoder(y_str) as dec:
        with tqdm(product(range(yH), range(yW)), ncols=80, total=yH*yW) as qbar:
            for h, w in qbar:
                y_mu, y_std, y_w = comp_model.contextmodel.get_gmm_params(y_hat_pad[:, :, h:h+3, w:w+5])
                qcdf = get_gmm_qcdf(samples, y_mu, y_std, y_w)
                
                for ch in range(yC):
                    symbol = dec.decode(1, qcdf[:, ch], is_normalized=False)[0]
                    y_hat_pad[0, ch, h+2, w+2] = symbol - offset
    
    y_hat = y_hat_pad[:, :, 2:, 2:yW+2]
    with torch.no_grad():
        reconstcution = comp_model.decoder(y_hat)
        reconstcution = reconstcution[:, :, :H, :W]
    return reconstcution

def encode(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="path to config json")
    parser.add_argument("-m", "--model", type=str, required=True, help="path to model_weight")
    parser.add_argument("-i", "--input", type=str, required=True, help="path to input image")
    parser.add_argument("-o", "--output", type=str, required=True, help="path to output bin")
    args = parser.parse_args(argv)
    opt = load_opt(args.config)
    opt.update(vars(args))
    model = load_model(opt)
    img = load_img(opt.input)
    _, bpp = compress_img(model, img, opt.output)
    print(f'{opt.input} -> {opt.output} : {bpp:.4}bpp')

def decode(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help='path to config json')
    parser.add_argument("-m", "--model", type=str, required=True, help="path to model_weight")
    parser.add_argument("-i", "--input", type=str, required=True, help="path to input bin file")
    parser.add_argument("-o", "--output", type=str, required=True, help="path to output image")
    parser.add_argument( "--interp_model", type=str, help="path to interpolate model")
    parser.add_argument( "--interp_alpha", type=float, default=1.0, help="alpha * model + (1-alpga) * interp_model")
    args = parser.parse_args(argv)
    opt = load_opt(args.config)
    opt.update(vars(args))
    model = load_model(opt)
    recon = decompress_img(model, opt.input, opt.bottleneck)
    cv2.imwrite(opt.output, img_torch2np(recon))
    print(f"reconstruction -> {opt.output}")

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["encode", "decode"])
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv[0:1])
    argv = argv[1:]
    if args.command == "encode":
        encode(argv)
    elif args.command == "decode":
        decode(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
