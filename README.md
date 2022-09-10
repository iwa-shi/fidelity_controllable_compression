# Fidelity-Controllable Extreme Image Compression with Generative Adversarial Networks

This repository is a PyTorch implementation of following paper:  
Fidelity-Controllable Extreme Image Compression with Generative Adversarial Networks  
ICPR2020 Accepted  [(arxiv)](https://arxiv.org/abs/2008.10314)  
Shoma Iwai, Tomo Miyazaki, Yoshihiro Sugaya, and Shinichiro Omachi

![](https://github.com/iwa-shi/fidelity_controllable_compression/blob/master/fig/others_compare_kodim21.png)

## Our Environment (updated 2022/09/10)
```
    Python==3.6.9
    pytorch==1.0.0
    scipy==1.3.2
    numpy==1.17.4
    opencv-python==4.1.1.26
    range-coder==1.1
    tqdm==4.38.0
```

## Pretrained Model
Download our pretrained [model](https://drive.google.com/file/d/1RHphLaixbcRq7-CQrYLwOlkoCXn7rCrs/view?usp=sharing) and unzip it.  
`ckpt_model*_mse.pth` is trained in the first stage, and `ckpt_model*_gan.pth` is fine-tuned in the second stage. These two models share the same encoder.

|  model name | Average bitrate (Kodak) |
| ------------- | :----------------------:|
| ckpt_model1_*.pth |  0.0300 bpp | 
| ckpt_model2_*.pth |  0.0624 bpp |

## Train
```
    python train.py ./config/default.json
```
Learned weights will be stored at `checkpoint/{exp}/model`.
Note that, `train.py` is a very simple training code with minimal functionality.

## Test (updated 2022/09/10)
#### Encoding
```
    python codec.py encode -c CONFIG_PATH -m MODEL_PATH -i IMAGE_PATH -o BIN_PATH
```
#### Decoding
```
    python codec.py decode -c CONFIG_PATH -m MODEL_PATH -i BIN_PATH -o SAVE_IMG_PATH
```
For example, 
```
    python codec.py encode -c config/default.json -m checkpoint/ckpt_model1_gan.pth -i images/kodim01.png -o outputs/binary/kodim01.pth
    python codec.py decode -c config/default.json -m checkpoint/ckpt_model1_gan.pth -i outputs/binary/kodim01.pth -o outputs/reconstruction/kodim01_recon.png
```

#### Network Interpolation
If you want to use network interpolation, specify `--interp_model` and `__interp_alpha`.
```
    python codec.py decode -c CONFIG_PATH -m MODEL_PATH -i BIN_PATH -o SAVE_IMG_PATH --interp_model INTERP_MODEL_PATH --interp_alpha 0.8
```
The interpolated weight will be `interp_alpha * torch.load(MODEL_PATH) + (1 - interp_alpha) * torch.load(INTERP_MODEL_PATH)`.
You can balance the trade-off between distortion and perception by changing alpha.
![](https://github.com/iwa-shi/fidelity_controllable_compression/blob/master/fig/interp_compare.png)


## Acknowledgments
We thank [Jorge Pessoa](https://github.com/jorge-pessoa) for the code of GDN.
