# Fidelity-Controllable Extreme Image Compression with Generative Adversarial Networks

This repository is a PyTorch implemention of following paper:  
Fidelity-Controllable Extreme Image Compression with Generative Adversarial Networks  
arxiv url (https://arxiv.org/abs/2008.10314)  
Shoma Iwai, Tomo Miyazaki, Yoshihiro Sugaya, and Shinichiro Omachi

![](https://github.com/iwa-shi/fidelity_controllable_compression/blob/master/fig/others_compare_kodim21.png)

## Our Environment
```
    Python==3.6.9
    pytorch==1.0.0
    scipy==1.3.2
    numpy==1.17.4
    opencv-python==4.1.1.26
    tqdm==4.38.0
```

## Pretrained Model
Download our pretrained [model](https://drive.google.com/file/d/1RHphLaixbcRq7-CQrYLwOlkoCXn7rCrs/view?usp=sharing) and unzip it.  
`ckpt_model*_mse.pth` is trained in the first stage, and `ckpt_model*_gan.pth` is fine-tuned in the second stage. These two models share the same encoder.

|  model name | Average bitrate (Kodak) |
| ------------- | :----------------------:|
| ckpt_model1_*.pth |  0.0299 bpp | 
| ckpt_model2_*.pth |  0.0623 bpp |


## Test 
```
    python compress.py MODEL_PATH IMAGE_PATH
```
Compressed files will be stored at `outputs/binary`.
```
    python decompress.py MODEL_PATH BINARY_PATH
```
For example, 
```
    python compress.py checkpoints/ckpt_model1_gan.pth images/
    python decompress.py checkpoints/ckpt_model1_gan.pth outputs/binary
```

#### Network Interpolation
If you want to use network interpolation, run `decompress_netinterp.py`.
```
    python decompress_netinterp.py MODEL_PATH MODEL_PATH2 ALPHA BINARY_PATH
```
For example, 
```
    python compress.py checkpoints/ckpt_model2_gan.pth images/
    python decompress_netinterp.py checkpoints/ckpt_model2_gan.pth checkpoints/ckpt_model2_mse.pth 0.8 outputs/binary
```
You can balance the trade-off between distortion and perception by changing alpha.
![](https://github.com/iwa-shi/fidelity_controllable_compression/blob/master/fig/interp_compare.png)


## Acknowledgments
We thank [nayuki](https://github.com/nayuki) and [Jorge Pessoa](https://github.com/jorge-pessoa) for the code of atirhmetic coding and GDN, respectively.