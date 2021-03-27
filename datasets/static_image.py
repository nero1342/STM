import os 
import numpy as np 
from PIL import Image 

import torch 
import torchvision 
from torch.utils import data 

import glob 
import albumentations as A

class StaticImage(data.Dataset):

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False, augmentation = None):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)
        
        
        self.augmentation = augmentation
        if self.augmentation is None:
            self.augmentation = get_training_augmentation() 

        self.frames = [] 
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                frames = glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))
                self.num_frames[_video] = len(frames)
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

                for it, frame in enumerate(frames):
                    self.frames.append((_video, it))

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.frames)

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video, index = self.frames[index]
        info = {}
        info['name'] = video
        info['num_frames'] = 3
        info['size_480p'] = (384, 384) # self.size_480p[video]#

        N_frames = np.empty((3,)+info['size_480p']+(3,), dtype=np.float32)
        N_masks = np.empty((3,)+info['size_480p'], dtype=np.uint8)
        
        
        img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(index))
        image = np.array(Image.open(img_file).convert('RGB'))
        mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(index))  
        mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        
        # apply augmentations
        for i in range(3):
            sample = self.augmentation(image=image, mask=mask)
            N_frames[i], N_masks[i] = sample['image'], sample['mask']
            N_frames[i] /= 255.

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=20, p = 1, border_mode = 0),

        # A.PadIfNeeded(min_height=480, min_width=864, always_apply=True,border_mode = 0),
        A.RandomCrop(height=384, width=384, always_apply=True),

        #A.IAAAdditiveGaussianNoise(p=0.2),
        #A.IAAPerspective(p=0.5),

        #A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(480, 864)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)
