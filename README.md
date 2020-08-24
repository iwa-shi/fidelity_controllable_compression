# Fidelity-Controllable Extreme Image Compression with Generative Adversarial Networks

This repository is a PyTorch implemention of following paper:  
Fidelity-Controllable Extreme Image Compression with Generative Adversarial Networks  
arxiv url (https://)  
Shoma Iwai, Tomo Miyazaki, Yoshihiro Sugaya, and Shinichiro Omachi


## Environment
```
    Python==3.6.9
    pytorch==1.0.0
    torchvision==0.2.1
    scipy==1.3.2
    numpy==1.17.4
    tqdm
```

## Test
Download our pretrained [model]() and unzip it.  
`ckpt_model1_mse.pth` is trained in the first stage, and `ckpt_model1_gan.pth` is fine-tuned in the second stage. The compression rate of `ckpt_model1_*.pth` is higher than `ckpt_model2_*.pth`.
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

![](https://github.com/iwa-shi/fidelity_controllable_compression/blob/master/fig/others_compare_kodim21.png)

#### Network Interpolation
If you want to use network interpolation, run decompress_netinterp.py.
```
    python decompress_netinterp.py MODEL_PATH MODEL_PATH2 ALPHA BINARY_PATH
```
For example, 
```
    python compress.py checkpoints/ckpt_model2_gan.pth images/
    python decompress.py checkpoints/ckpt_model2_gan.pth checkpoints/ckpt_model2_mse.pth 0.8 outputs/binary
```
You can balance the trade-off between distortion and perception by changing alpha.
![](https://github.com/iwa-shi/fidelity_controllable_compression/blob/master/fig/interp_compare.png)
