import torch.utils.data as torch_data
from torchvision import transforms
from PIL import Image
from copy import deepcopy
import os
import numpy as np


class CELEB_A_HQ(torch_data.Dataset):
    def __init__(self,
                 dataset,
                 mode="train",
                 transform=transforms.ToTensor(),
                 data_root=''):
        '''
        targets: list of values for classification
        or list of paths to segmentation mask for segmentation task.
        augment: list of keywords for augmentations.
        '''
        self.dataset = dataset
        self.mode = mode
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        source_image = np.array(Image.open(os.path.join(self.data_root, self.dataset.iloc[[index]]['path'].values[0])))
        target_image = deepcopy(source_image)
        if self.mode == 'train':
            mid = int((self.dataset.iloc[[index]]['28'].values[0] + self.dataset.iloc[[index]]['4'].values[0]) / 2)

            width = int(abs(self.dataset.iloc[[index]]['28'].values[0] - self.dataset.iloc[[index]]['4'].values[0]))
            height = width * 0.77
            left_x = int((self.dataset.iloc[[index]]['56'].values[0] + mid) / 2 - width // 2)
            left_y = int(self.dataset.iloc[[index]]['57'].values[0])

            bbx1, bby1, bbx2, bby2 = self.randbbox(width, height, lam=0)

            bbx1 = int(bbx1 + left_x)
            bbx2 = int(bbx2 + left_x)
            bby1 = int(bby1 + left_y)
            bby2 = int(bby2 + left_y)

            if bbx1 > bbx2:
                bbx1, bbx2 = bbx2, bbx1
            if bby1 > bby2:
                bby1, bby2 = bby2, bby1
            
            source_image[bby1:bby2, bbx1:bbx2] = np.random.randint(low=0, high=255, size=source_image[bby1:bby2, bbx1:bbx2].shape)

        else:
            mid = int((self.dataset.iloc[[index]]['28'].values[0] + self.dataset.iloc[[index]]['4'].values[0]) / 2)

            width = abs(self.dataset.iloc[[index]]['28'].values[0] - self.dataset.iloc[[index]]['4'].values[0]) * 0.95
            height = width * 0.78
            left_x = int((self.dataset.iloc[[index]]['56'].values[0] + mid) / 2 - width // 2)
            left_y = int(self.dataset.iloc[[index]]['57'].values[0])
            
            source_image[left_y:int(left_y + height), left_x:int(left_x + width)] = np.random.randint(low=0, high=255, size=source_image[left_y:int(left_y + height), left_x:int(left_x + width)].shape)

        source_image = Image.fromarray(source_image)
        target_image = Image.fromarray(target_image)

        source = self.transform(source_image)
        target = self.transform(target_image)

        return source, target

    def randbbox(self, width, height, lam):
        W = width
        H = height

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        #cx = np.random.randint(W)
        #cy = np.random.randint(H)
        alpha = 80.0
        beta = 80.0
        cx = int(W * np.random.beta(alpha, beta))
        cy = int(H * np.random.beta(alpha, beta))

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2