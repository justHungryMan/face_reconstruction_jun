import os
import sys
import logging
import datetime
import random
import numpy as np
import copy
import argparse
from contextlib import suppress

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel

import torchvision
import torchvision.transforms as transforms

import hydra
from omegaconf import DictConfig, OmegaConf

import trainer

from tqdm import tqdm 
import wandb

class Trainer():
    def __init__(self, conf, rank=0):
        self.conf = copy.deepcopy(conf)
        self.rank = rank
        self.is_master = True if rank == 0 else False
        self.alpha = self.conf.dataset.alpha
        self.set_env()

    def set_env(self):
        if self.is_master:
            wandb.init(config=self.conf, project='face_reconstruction')
            wandb.run.name = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(self.rank)


        # mixed precision
        self.amp_autocast = suppress
        if self.conf.base.use_amp is True:
            self.amp_autocast = torch.cuda.amp.autocast
            self.G_scaler = torch.cuda.amp.GradScaler(enabled=True)
            self.D_scaler = torch.cuda.amp.GradScaler(enabled=True)
            if self.is_master:
                print(f'[Hyper]: Use Mixed precision - float16')
        else:
            self.scaler = None
        
        # Load checkpoint
        self.start_epoch = 1


    def build_model(self, num_classes=-1):
        G, D = trainer.architecture.create(self.conf.architecture)
        G = G.to(device=self.rank, non_blocking=True)
        D = D.to(device=self.rank, non_blocking=True)
        G = DDP(G, device_ids=[self.rank], output_device=self.rank)
        D = DDP(D, device_ids=[self.rank], output_device=self.rank)

        return G, D

    def build_optimizer(self, model):
        optimizer = trainer.optimizer.create(self.conf.optimizer, model)
        return optimizer


    def build_scheduler(self, optimizer):
        scheduler = trainer.scheduler.create(self.conf.scheduler, optimizer)

        return scheduler
    # TODO: modulizaing
    def build_dataloader(self, ):
        data = pd.read_csv(os.path.join(self.conf.dataset.path, 'landmark.csv'))
        
        train_data, test_data = train_test_split(data, test_size=0.025, random_state=42)
        train_data.reset_index(drop=True, inplace=True)
        train_data = train_data.rename_axis('index').reset_index()
        test_data.reset_index(drop=True, inplace=True)
        test_data = test_data.rename_axis('index').reset_index()

        train_loader, train_sampler = trainer.dataset.create(
            self.conf.dataset,
            dataset=train_data,
            world_size=self.conf.base.world_size,
            local_rank=self.rank,
            mode='train'
        )
        test_loader, test_sampler = trainer.dataset.create(
            self.conf.dataset,
            dataset=test_data,
            world_size=self.conf.base.world_size,
            local_rank=self.rank,
            mode='test'
        )


        return train_loader, train_sampler, test_loader, test_sampler

    def build_loss(self):
        criterion = trainer.loss.create(self.conf.loss, self.rank)
        criterion.to(device=self.rank, non_blocking=True)

        return criterion

    
    def load_model(self, model, path):
        data = torch.load(path)
        key = 'model' if 'model' in data else 'state_dict'

        if not isinstance(model, (DataParallel, DDP)):
            model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
        else:
            model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})
        return model

    def patch_loss(self, criterion, input, TF):
        if TF is True:
            comparison = torch.ones_like(input)
        else:
            comparison = torch.zeros_like(input)
        return criterion(input, comparison)

    def train_one_epoch(self, epoch, G, D, dl, criterion, L1_distance, G_optimizer, D_optimizer):
        # for step, (image, label) in tqdm(enumerate(dl), total=len(dl), desc="[Train] |{:3d}e".format(epoch), disable=not flags.is_master):
        train_hit = 0
        train_total = 0
        one_epoch_loss = 0
        # 0: G_loss, 1: D_loss, 2: len(dl)
        counter = torch.zeros((3, ), device=self.rank)

        G.train()
        D.train()
        pbar = tqdm(
            enumerate(dl), 
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(dl), 
            desc=f"train:{epoch}/{self.conf.dataset.epochs}", 
            disable=not self.is_master
            )
        for step, (source, target) in pbar:
            
            # current_step = epoch*len(dl)+step
            current_step = epoch
            source = source.to(device=self.rank, non_blocking=True)
            target = target.to(device=self.rank, non_blocking=True)

            with self.amp_autocast():
                fake_image = G(source)
                D_loss = self.patch_loss(criterion, D(target), True) + self.patch_loss(criterion, D(fake_image), False)
                G_loss = self.patch_loss(criterion, D(fake_image), True) + L1_distance(target, fake_image) * self.alpha

            if self.G_scaler is None or self.D_scaler is None:
                D_optimizer.zero_grad(set_to_none=True)
                D_loss.backward()
                D_optimizer.step()

                G_optimizer.zero_grad(set_to_none=True)
                G_loss.backward()
                G_optimizer.step()
            else:
                D_optimizer.zero_grad(set_to_none=True)
                self.D_scaler.scale(D_loss).backward(retain_graph=True)
                self.D_scaler.step(D_optimizer)
                self.D_scaler.update()

                G_optimizer.zero_grad(set_to_none=True)
                self.G_scaler.scale(G_loss).backward()
                self.G_scaler.step(G_optimizer)
                self.G_scaler.update()
            
                

            counter[0] += G_loss.item()
            counter[1] += D_loss.item()
            if step % 100 == 1:
                pbar.set_postfix({'G_loss': round(G_loss.item(), 3),'D_Loss':round(D_loss.item(), 3)}) 

        counter[2] += len(dl)
        torch.distributed.reduce(counter, 0)
        if self.is_master:
            counter = counter.detach().cpu().numpy()
            G_loss = counter[0] / counter[2]
            D_loss = counter[1] / counter[2]
            images1 = wandb.Image(torchvision.utils.make_grid([source[0], fake_image[0], target[0]]), caption='Top: Input Image, Mid: Generated Image, Bottom: Ground Truth')
            images2 = wandb.Image(torchvision.utils.make_grid([source[1], fake_image[1], target[1]]), caption='Top: Input Image, Mid: Generated Image, Bottom: Ground Truth')
            images3 = wandb.Image(torchvision.utils.make_grid([source[2], fake_image[2], target[2]]), caption='Top: Input Image, Mid: Generated Image, Bottom: Ground Truth')
            images4 = wandb.Image(torchvision.utils.make_grid([source[3], fake_image[3], target[3]]), caption='Top: Input Image, Mid: Generated Image, Bottom: Ground Truth')

            wandb.log({
                'train/examples1': images1,
                'train/examples2': images2,
                'train/examples3': images3,
                'train/examples4': images4
            }, step=epoch
            )


        
        # return loss, accuracy
        return G_loss, D_loss


    @torch.no_grad()
    def eval(self, epoch, G, D, dl, criterion, L1_distance):
        # 0: val_loss, 1: val_hit, 2: val_total, 3: len(dl)
        counter = torch.zeros((4, ), device=self.rank)
        G.eval()
        D.eval()
        pbar = tqdm(
            enumerate(dl),
            bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
            total=len(dl),
            desc=f"val  :{epoch}/{self.conf.dataset.epochs}", 
            disable=not self.is_master
            ) # set progress bar

        for step, (source, target) in pbar:
            # current_step = epoch*len(dl)+step
            current_step = epoch
            source = source.to(device=self.rank, non_blocking=True)
            target = target.to(device=self.rank, non_blocking=True)
            with self.amp_autocast():
                fake_image = G(source)
                D_loss = self.patch_loss(criterion, D(target), True) + self.patch_loss(criterion, D(fake_image), False)
                G_loss = self.patch_loss(criterion, D(fake_image), True) + L1_distance(target, fake_image) * self.alpha


            counter[0] += G_loss.item()
            counter[1] += D_loss.item()
            if step % 100 == 1:
                pbar.set_postfix({'G_loss': round(G_loss.item(), 3),'D_Loss':round(D_loss.item(), 3)}) 

        counter[2] += len(dl)
        torch.distributed.reduce(counter, 0)
        if self.is_master:
            counter = counter.detach().cpu().numpy()
            G_loss = counter[0] / counter[2]
            D_loss = counter[1] / counter[2]
            
            images1 = wandb.Image(torchvision.utils.make_grid([source[0], fake_image[0], target[0]]), caption='Top: Input Image, Mid: Generated Image, Bottom: Ground Truth')
            images2 = wandb.Image(torchvision.utils.make_grid([source[1], fake_image[1], target[1]]), caption='Top: Input Image, Mid: Generated Image, Bottom: Ground Truth')
            images3 = wandb.Image(torchvision.utils.make_grid([source[2], fake_image[2], target[2]]), caption='Top: Input Image, Mid: Generated Image, Bottom: Ground Truth')
            images4 = wandb.Image(torchvision.utils.make_grid([source[3], fake_image[3], target[3]]), caption='Top: Input Image, Mid: Generated Image, Bottom: Ground Truth')


            wandb.log({
                'val/examples1': images1,
                'val/examples2': images2,
                'val/examples3': images3,
                'val/examples4': images4
            }, step=epoch
            )

        
        # return loss, accuracy
        return G_loss, D_loss

    def train_eval(self):
        G, D = self.build_model()
        criterion = nn.MSELoss().to(device=self.rank, non_blocking=True)
        L1_distance = nn.L1Loss().to(device=self.rank, non_blocking=True)

        G_optimizer = self.build_optimizer(G)
        D_optimizer = self.build_optimizer(D)

        G_scheduler = self.build_scheduler(G_optimizer)
        D_scheduler = self.build_scheduler(D_optimizer)

        train_dl, train_sampler, test_dl, _ = self.build_dataloader()

        # Wrap the model
        
        # initialize
        for name, x in G.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

        for name, x in D.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()
        
        # load checkpoint
        print("TRAINING START")
        for epoch in range(self.start_epoch,  self.conf.dataset.epochs + 1):
            train_sampler.set_epoch(epoch)

            # train
            train_G_loss, train_D_loss = self.train_one_epoch(epoch, G, D, train_dl, criterion, L1_distance, G_optimizer, D_optimizer)
            G_scheduler.step()
            D_scheduler.step()

            # eval
            val_G_loss, val_D_loss = self.eval(epoch, G, D, test_dl, criterion, L1_distance)
            
            torch.cuda.synchronize()
            # save_model

            if self.is_master:
                print(f'Epoch {epoch}/{self.conf.dataset.epochs} - train/G_loss: {train_G_loss:.3f}, train/D_loss: {train_D_loss:.3f}, val/G_loss: {val_G_loss:.3f}, val/D_loss: {val_D_loss:.3f}')
                wandb.log({
                    'epoch': epoch,
                    'G_lr': G_scheduler.get_last_lr(),
                    'D_lr': D_scheduler.get_last_lr(), 
                    'train/G_loss': train_G_loss,
                    'train/D_loss': train_D_loss,
                    'val/G_loss': val_G_loss,
                    'val/D_loss': val_D_loss,
                }, step=epoch
                )

    def run(self):
        if self.conf.base.mode == 'train':
            pass
        elif self.conf.base.mode == 'train_eval':
            self.train_eval()
        elif self.conf.base.mode == 'finetuning':
            pass

def set_seed(conf):
    if conf.base.seed is not None:
        conf.base.seed = int(conf.base.seed, 0)
        print(f'[Seed] :{conf.base.seed}')
        os.environ['PYTHONHASHSEED'] = str(conf.base.seed)
        random.seed(conf.base.seed)
        np.random.seed(conf.base.seed)
        torch.manual_seed(conf.base.seed)
        torch.cuda.manual_seed(conf.base.seed)
        torch.cuda.manual_seed_all(conf.base.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True

def runner(rank, conf):
    # Set Seed
    set_seed(conf)

    os.environ['MASTER_ADDR'] = conf.MASTER_ADDR
    os.environ['MASTER_PORT'] = conf.MASTER_PORT

    print(f'Starting train method on rank: {rank}')
    dist.init_process_group(
        backend='nccl', world_size=conf.base.world_size, init_method='env://',
        rank=rank
    )
    trainer = Trainer(conf, rank)
    trainer.run()
    wandb.finish()

@hydra.main(config_path='conf', config_name='default')
def main(conf: DictConfig) -> None:
    print(f'Configuration\n{OmegaConf.to_yaml(conf)}')
    
    mp.spawn(runner, nprocs=conf.base.world_size, args=(conf, ))
    

if __name__ == '__main__':
    main()