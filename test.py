import os
import torch
from torch.utils.data import DataLoader
from model.model import CompModel
from opt import opt_test
from evaluation import eval_model
import torchvision.transforms as T
from glob import glob


def load_img(path):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            img = transform(img)

    _, _, h, w = img.size()
    h_, w_ = h, w
    if h % 16 != 0:
        h_ = (h // 16 + 1) * 16
    if w % 16 != 0:
        w_ = (w // 16 + 1) * 16
    x = torch.zeros((1, 3, h_, w_))
    x[:, :, :h, :w] = img

    return x

def main(args):
    device = args.device

    

    comp_model = CompModel(args).to(device)
    comp_model.eval()
    state_dict = torch.load(args.model_path, map_location=device)
    comp_model.load_state_dict(state_dict)
    print('load model')
    
    df = eval_model(comp_model, dataloader, device=device, save_img=True, save_dir=args.sample_dir, 
                color_space=args.color_space, use_seg_mask=args.use_seg_mask, 
                mask_size=args.mask_size, input_seg=args.input_seg)
    print(df.loc['mean'])
    df.to_csv(os.path.join(args.sample_dir, 'data.csv'))
    

if __name__ == "__main__":
    args = opt_test()
    main(args)
