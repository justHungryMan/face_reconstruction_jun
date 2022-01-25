import torch.nn as nn
import logging

LOGGER = logging.getLogger(__name__)

def create(conf, rank):

    if conf['type'] == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        raise AttributeError(f'not support loss config: {conf}')

    return criterion