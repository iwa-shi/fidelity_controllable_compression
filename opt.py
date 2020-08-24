import os
import json
import argparse

def opt_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp')
    parser.add_argument('image_dir')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--resume_training', action='store_true')

    # training setting
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--iter1', type=int, default=500000)
    parser.add_argument('--iter2', type=int, default=300000)
    
    # learning rate
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--lr_half_step', type=int, default=150000)
    parser.add_argument('--d_lr', type=float, default=0.00003)
    parser.add_argument('--d_lr_half_step', type=int, default=150000)
    
    # step
    parser.add_argument('--save_step', type=int, default=50000)
    parser.add_argument('--log_step', type=int, default=100)

    # loss weight
    parser.add_argument('--lamb_mse1', type=float, default=0.5)
    parser.add_argument('--lamb_mse2', type=float, default=0.5)
    parser.add_argument('--lamb_adv', type=float, default=0.05)
    parser.add_argument('--lamb_vgg', type=float, default=1)

    # model hyper parameter
    parser.add_argument('--bottleneck', type=int, default=32)
    parser.add_argument('--main_channel', type=int, default=192)
    parser.add_argument('--gmm_K', type=int, default=3)

    args = parser.parse_args()
    args.model_dir = os.path.join('checkpoint', args.exp, 'model')
    os.makedirs(args.model_dir, exist_ok=True)
    
    with open(os.path.join('checkpoint', args.exp, 'settings.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, separators=(',', ': '))

    return args


def opt_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    
    parser.add_argument('--image_path')
    parser.add_argument('--binary_path')
    
    parser.add_argument('--bottleneck', type=int, default=32)
    parser.add_argument('--main_channel', type=int, default=192)
    parser.add_argument('--gmm_K', type=int, default=3)

    parser.add_argument('--use_net_interp', action='store_true')
    parser.add_argument('--interp_alpha', type=float)
    parser.add_argument('--model_path2', type=str)
    
    args = parser.parse_args()

    return args
