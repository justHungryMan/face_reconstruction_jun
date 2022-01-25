import torch
import logging

from . import unet
from . import discriminator
LOGGER = logging.getLogger(__name__)

def create(conf, num_classes=None):
    base, architecture_name = [l.lower() for l in conf['type'].split('/')]

    if base == 'unet':
        architecture = unet.UnetGenerator(**conf['params'])
    else:
        raise AttributeError(f'not support architecture config: {conf}')
    D = discriminator.Discriminator()

    return architecture, D