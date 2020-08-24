import argparse
from collections import OrderedDict

#import numpy as np
#import os

import torch

from model.model import CompModel
from decompress import decompress

def load_model(args):
    args.device = 'cpu'
    comp_model = CompModel(args)

    # use network interpolation
    state_dict1 = torch.load(args.model_path, map_location='cpu')['comp_model']
    state_dict2 = torch.load(args.model_path2, map_location='cpu')['comp_model']
    state_dict_interp = OrderedDict()
    for k in state_dict1:
        if 'decoder' in k:
            state_dict_interp[k] = args.alpha * state_dict1[k] + (1 - args.alpha) * state_dict2[k]
        else:
            state_dict_interp[k] = state_dict1[k]
    comp_model.load_state_dict(state_dict_interp)

    comp_model.eval()
    return comp_model
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('model_path2')
    parser.add_argument('alpha', type=float)
    parser.add_argument('binary_path')
    
    parser.add_argument('--bottleneck', type=int, default=32)
    parser.add_argument('--main_channel', type=int, default=192)
    parser.add_argument('--gmm_K', type=int, default=3)

    args = parser.parse_args()
    comp_model = load_model(args)
    decompress(comp_model, args)