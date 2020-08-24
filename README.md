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

```
    python compress.py MODEL_PATH IMAGE_PATH
```
Compressed files will be stored at `outputs/binary`.
```
    python decompress.py MODEL_PATH BINARY_PATH
```

### Network Interpolation
If you want to use network interpolation, run decompress_netinterp.py.
```
    python decompress_netinterp.py MODEL_PATH MODEL_PATH2 ALPHA BINARY_PATH
```
