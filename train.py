import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from addict import Dict
from torch.utils.data import Dataset
from tqdm import tqdm

from model.discriminator import GANLoss, MultiscaleDiscriminator
from model.model import CompModel
from model.perceptual_loss import VGGLoss_ESRGAN


class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = os.listdir(self.image_dir)
        transform = [
            T.RandomCrop((256, 256)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        self.transform = T.Compose(transform)
        self.resize = T.Resize(256)
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, id):
        filename = self.image_list[id]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                if min(image.size) < 256:
                    image = self.resize(image)
                image = self.transform(image.convert('RGB'))
        return image

def opt_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='path to config json')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        opt = json.load(f)
    opt = Dict(opt)
    opt.model_dir = os.path.join(opt.checkpoint_dir, opt.exp, 'model')
    os.makedirs(opt.model_dir, exist_ok=True)

    with open(os.path.join(opt.checkpoint_dir, opt.exp, 'config.json'), 'w') as f:
        json.dump(opt, f, indent=2, separators=(',', ': '))

    return opt


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.set_models()
        self.set_dataloader()
        self.set_loss()

    def set_models(self):
        self.comp_model = CompModel(self.opt).to(self.device)
        self.discriminator = MultiscaleDiscriminator(input_nc=3, getIntermFeat=False).to(self.device)

    def set_optimizer_scheduler(self, is_first_stage: bool):
        if is_first_stage:
            self.optimizer = torch.optim.Adam(self.comp_model.parameters(), lr=self.opt.lr, betas=(0, 0.999))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.opt.lr_half_step, gamma=0.5)
        else:
            self.optimizer = torch.optim.Adam(self.comp_model.decoder.parameters(), lr=self.opt.lr, betas=(0, 0.999))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.opt.lr_half_step, gamma=0.5)
            self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.d_lr, betas=(0, 0.999))
            self.d_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.d_optimizer, milestones=self.opt.d_lr_half_step, gamma=0.5)

    def set_loss(self):
        self.mse_loss = torch.nn.MSELoss().to(self.device)
        self.vgg_loss = VGGLoss_ESRGAN().to(self.device)
        self.adv_loss = GANLoss()

    def set_dataloader(self):
        dataset = ImageDataset(self.opt.image_dir)
        self.dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.opt.batch_size,
                drop_last=True, shuffle=True, num_workers=8)

    def save_checkpoint(self, current_itr, save_path, is_second_stage=False):
        state = {
            'iter':current_itr,
            'comp_model':self.comp_model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict()
        }
        
        if is_second_stage:
            state['discriminator'] = self.discriminator.state_dict()
            state['d_optimizer'] = self.d_optimizer.state_dict()
            state['d_scheduler'] = self.d_scheduler.state_dict()
        torch.save(state, save_path)

    def run(self):
        self.set_optimizer_scheduler(is_first_stage=True)
        self.train_loop_stage1()
        self.set_optimizer_scheduler(is_first_stage=False)
        self.train_loop_stage2()

    def train_data_generator(self, num_iter):
        data_iter = iter(self.dataloader)
        for i in tqdm(range(num_iter), ncols=100):
            try:
                real_images = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                real_images = next(data_iter)
            yield i+1, real_images

    def train_loop_stage1(self):
        for itr, real_images in self.train_data_generator(self.opt.total_iter1):
            real_images = real_images.to(self.device)
            
            self.optimizer.zero_grad()
            fake_images, bpp = self.comp_model(real_images)
            real255 = (real_images + 1.) * 255. / 2.
            fake255 = (fake_images + 1.) * 255. / 2.

            mse = self.mse_loss(fake255, real255)
            loss = bpp + self.opt.lamb_mse1 * mse / 1000.

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # save checkpoint
            if itr % self.opt.save_step == 0 or itr == self.opt.total_iter1:
                save_path = os.path.join(self.opt.model_dir, 'ckpt_stage1.pth.tar')
                self.save_checkpoint(itr, save_path)

            # print log
            if itr % self.opt.print_step == 0:
                print('\nstage1 iter %7d : distortion %.3f  rate %.3f  total_loss %.3f' % (itr, mse, bpp, loss.item()))

    def train_loop_stage2(self):
        for itr, real_images in self.train_data_generator(self.opt.total_iter2):
            real_images = real_images.to(self.device)
            # train comp_model
            self.optimizer.zero_grad()
            fake_images = self.comp_model.train_only_decoder(real_images)

            real255 = (real_images + 1.) * 255. / 2.
            fake255 = (fake_images + 1.) * 255. / 2.

            mse = self.mse_loss(fake255, real255)
            vgg = self.vgg_loss(real_images, fake_images)
            d_f = self.discriminator(fake_images)
            adv = self.adv_loss(d_f, True)
            loss = self.opt.lamb_mse2 * mse / 1000. + self.opt.lamb_vgg * vgg + self.opt.lamb_adv * adv
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # train discriminator
            self.d_optimizer.zero_grad()
            d_r = self.discriminator(real_images)
            d_f = self.discriminator(fake_images.detach())
            d_loss_real = self.adv_loss(d_r, True)
            d_loss_fake = self.adv_loss(d_f, False)
            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            self.d_optimizer.step()
            self.d_scheduler.step()

            # save checkpoint
            if itr % self.opt.save_step == 0 or itr == self.opt.total_iter2:
                save_path = os.path.join(self.opt.model_dir, 'ckpt_stage2.pth.tar')
                self.save_checkpoint(itr, save_path, is_second_stage=True)

            # print log
            if itr % self.opt.print_step == 0:
                print('\nstage2 iter %7d : distortion %.3f vgg %.3f adv_G %.3f adv_D %.3f' % 
                        (itr, mse, vgg, adv, d_loss))


def main():
    opt = opt_train()
    trainer = Trainer(opt)
    trainer.run()
    
if __name__ == '__main__':
    main()
