import torch
import torchvision
import torchvision.transforms as transforms
import logging
import os
import pandas as pd
from .celeba_hq import CELEB_A_HQ
from .data_utils import preprocess


LOGGER = logging.getLogger(__name__)


def create(conf, dataset, world_size=1, local_rank=-1, mode='train'):
    data_path = conf['path']
    conf = conf[mode]
    transformers = transforms.Compose([preprocess(t) for t in conf['preprocess']] )
    
    if conf['name'] == 'celeba_hq':
        dataset = CELEB_A_HQ(dataset=dataset,
                                 mode=mode,
                                 transform=transformers,
                                 data_root=data_path
                                 )

    else:
        raise AttributeError(f'not support dataset config: {conf}')
    
    sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=local_rank,
            shuffle=(mode == 'train')
        )
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=conf['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=conf['drop_last'],
        num_workers=os.cpu_count() // world_size - 1,
        sampler=sampler,
        persistent_workers=True
    )

    return dl, sampler