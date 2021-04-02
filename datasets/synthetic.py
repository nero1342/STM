import os 
import numpy as np 
from PIL import Image 
try:
    from utils import getter 
except:
    pass 
try:
    from .utils import getter 
except:
    pass 
    
import torch 
from torchvision  import transforms as tvtf 

from torch.utils import data 

import glob 
import albumentations as A
import random 

class SyntheticDataset:
    def __init__(self, dataset, niters, nimgs):
        self.dataset = getter.get_instance(dataset) 
        self.niters = niters
        self.nimgs = nimgs
        
        if len(self.dataset) != len(self):
            print("Random getitem")
        else:
            print("Get all items")
    def __len__(self):
        return self.niters

    def __getitem__(self, i):
        if len(self.dataset) != len(self):
          i = random.randrange(0, len(self.dataset))
        #i = i % (len(self.dataset))
        img, mask = self.dataset[i]
        
        imgs, masks = zip(*[self._augmentation(img, mask) for _ in range(self.nimgs)]) 
        im_0, *imgs = map(tvtf.ToTensor(), imgs)

        def mask2tensor(x): return torch.LongTensor(np.array(x)) 
        masks = list(map(mask2tensor, masks))
        mask_0, *masks = self._filter(masks) 
        

        num_objects = torch.max(mask_0)
        if num_objects == 0:
            return self.__getitem__(random.randrange(0, len(self.dataset)))
        return (im_0, mask_0, *imgs, num_objects), tuple(masks) 

    #@staticmethod
    def _augmentation(self, img, mask):
        train_transform = [
            A.PadIfNeeded(min_height=480, min_width=480, always_apply=True,border_mode = 0),
            A.Resize(480, 480),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, p = 1, border_mode = 0),
            A.RandomCrop(height=384, width=384, always_apply=True),
        ]
        transform = A.Compose(train_transform)
        result = transform(image = np.array(img), mask = np.array(mask))
        img, mask = result['image'], result['mask']
        return img, mask 
    
    def _filter_small_objs(self, mask, thres):
        # Filter small objects
        ori_objs = np.unique(mask)
        for obj in ori_objs:
            area = (mask == obj).sum().item()
            if area < thres:
                mask[mask == obj] = 0
        return mask
        
    def _filter_excessive_objs(self, masks):
        # Filter excessive objects
        ori_objs = np.unique(masks[0])
        for i in range(1, len(masks)):
            mask_objs = np.unique(masks[i])
            excess_objs = np.setdiff1d(mask_objs, ori_objs)
            for obj in excess_objs:
                masks[i][masks[i] == obj] = 0
        return masks

    def _filter(self, masks):
        #return masks
        masks[0] = self._filter_small_objs(masks[0], 1000)
        masks = self._filter_excessive_objs(masks)
        return masks