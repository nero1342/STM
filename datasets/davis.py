import torch
from torch.utils import data
from torchvision import transforms as tvtf
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from itertools import permutations
from pathlib import Path
from enum import Enum
import os
import random
import json


class DAVISCoreDataset(data.Dataset):
    def __init__(self, root_path=None,
                 annotation_folder="Annotations",
                 jpeg_folder="JPEGImages",
                 resolution="480p",
                 imageset_folder="ImageSets",
                 year="2017",
                 phase="train",
                 is_train=True,
                 mode=0,
                 min_skip=1,
                 max_skip=-1):
        super().__init__()

        # Root directory
        assert root_path is not None, "Missing root path, should be a path DAVIS dataset!"
        self.root_path = Path(root_path)

        self.annotation_folder = annotation_folder
        self.jpeg_folder = jpeg_folder
        self.imageset_folder = imageset_folder
        self.resolution = resolution

        self.is_train = is_train

        self.mode = mode
        self.min_skip = min_skip
        self.max_skip = max_skip

        # Path to Annotations
        self.annotation_path = self.root_path / self.annotation_folder / self.resolution

        # Load video name prefixes (ex: bear for bear_1)
        txt_path = \
            self.root_path / self.imageset_folder / str(year) / f"{phase}.txt"
        with open(txt_path) as files:
            video_name_prefixes = [filename.strip() for filename in files]

        # Load only the names that has prefix in video_name_prefixes
        self.video_names = [folder.name
                            for folder in self.annotation_path.iterdir()
                            if folder.name.split('_')[0] in video_name_prefixes]

        # Load video infos
        self.infos = self._load_videos_info(self.video_names)

    def _load_videos_info(self, video_names):
        infos = {}
        tbar = tqdm(video_names)
        for video_name in tbar:
            tbar.set_description_str(video_name)

            folder = self.annotation_path / video_name
            video_id = video_name.split('_')[0]

            info = dict()
            info['name'] = folder.name

            # Get video length
            jpeg_path = self.root_path / self.jpeg_folder / self.resolution / video_name
            info['length'] = len(list(jpeg_path.iterdir()))

            # Get total number of objects
            # YoutubeVOS provides a meta.json file
            if os.path.exists(str(self.root_path / 'meta.json')):
                json_data = json.load(
                    open(str(self.root_path / 'meta.json')))
                nobjects = len(json_data['videos'][video_id]['objects'])
                for x in folder.iterdir():
                    if ("ipynb_checkpoints" in str(x)):
                        continue
                    anno_im = Image.open(str(x)).convert('P')
                    break
            # Others might not, load all files just in case
            else:
                nobjects = 0
                for x in sorted(folder.iterdir()):
                    if ("ipynb_checkpoints" in str(x)):
                        continue
                    
                    anno_im = Image.open(str(x)).convert('P')
                    nobjects = max(nobjects, np.max(anno_im))
                    break
            info['nobjects'] = nobjects

            # Get image size (same for all frames)
            info['size'] = torch.tensor(anno_im.size)

            infos[video_name] = info
        return infos

    def _load_frame(self, img_name, augmentation):
        # Load annotated mask
        anno_path = str(self.annotation_path / img_name)
        mask = Image.open(anno_path).convert('P')

        # Load frame image
        jpeg_path = anno_path.replace(self.annotation_folder, self.jpeg_folder)
        jpeg_path = jpeg_path.replace('.png', '.jpg')
        img = Image.open(jpeg_path).convert('RGB')

        # Augmentation (if train)
        # if self.is_train:
        img, mask = augmentation(img, mask)
        
        # Convert to tensor
        img = tvtf.ToTensor()(img)
        mask = np.array(mask)
        # mask = torch.LongTensor(np.array(mask))

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

    def _filter(self, masks, small_obj_thres=1000):
        masks[0] = self._filter_small_objs(masks[0], small_obj_thres)
        masks = self._filter_excessive_objs(masks)
        return masks

    def _augmentation(self, img, mask):
        #img, mask = MultiRandomResize(resize_value=480)((img, mask))
        img = tvtf.Resize(384)(img)
        mask = tvtf.Resize(384)(mask)
        #img, mask = MultiRandomCrop(size=384)((img, mask))
        #img, mask = MultiRandomAffine(degrees=(-15, 15),
        #                              scale=(0.95, 1.05),
        #                              shear=(-10, 10))((img, mask))
        return img, mask
    
    def _augmentation_val(self, img, mask):
        #img, mask = MultiRandomResize(resize_value=480)((img, mask))
        img = tvtf.Resize(384)(img)
        mask = tvtf.Resize(384, 0)(mask)
        #img, mask = MultiRandomCrop(size=384)((img, mask))
        #img, mask = MultiRandomAffine(degrees=(-15, 15),
        #                              scale=(0.95, 1.05),
        #                              shear=(-10, 10))((img, mask))
        return img, mask


class DAVISPairDataset(DAVISCoreDataset):
    def __init__(self, max_npairs=-1, **kwargs):
        super().__init__(**kwargs)

        self.max_npairs = max_npairs

        # Generate frames
        self.frame_list = []
        for video_name in self.video_names:
            nobjects = self.infos[video_name]['nobjects']
            png_pair = self.get_frame(self.mode, video_name)
            for pair in png_pair:
                support_anno = video_name + "/" + pair[0]
                query_anno = video_name + "/" + pair[1]
                self.frame_list.append((support_anno, query_anno, nobjects))

    def get_frame(self, mode, video_name):
        images = sorted(os.listdir(str(self.annotation_path / video_name)))
        n = len(images)
        min_skip = self.min_skip
        max_skip = min(n - 1, self.max_skip if self.max_skip != -1 else n - 1)

        if mode == 0:
            return list(permutations(images, 2))
        elif mode == 1:
            return [(images[0], images[i]) for i in range(1, n) if max_skip >= i >= min_skip]
        elif mode == 2:
            indices = [(i, j) for i in range(n-1)
                       for j in range(i+1, n) if max_skip >= j - i >= min_skip]
            max_npairs = min(len(indices),
                             self.max_npairs if self.max_npairs != -1 else len(indices))
            indices = random.sample(indices, k=max_npairs)
            return [(images[i], images[j]) for i, j in indices]
        else:
            raise Exception('Unknown mode')

    def _get_mask(self, mask, anns):
        new_mask = np.zeros(mask.shape, dtype=np.int8)
        for i, ann in enumerate(anns):
            new_mask[mask == ann] = i + 1
        new_mask = torch.LongTensor(np.array(new_mask))

        return new_mask

    def __getitem__(self, inx):
        support_anno_name, query_anno_name, nobjects = self.frame_list[inx]

        ref_img, ref_mask = self._load_frame(support_anno_name,
                                             self._augmentation)
        query_img, query_mask = self._load_frame(query_anno_name,
                                                 self._augmentation)

        mask = np.array(ref_mask)
        annIds = list(set(list(mask.reshape(-1))))[1:]
        max_annot = 3
        choice = random.sample(annIds, k = min(max_annot, len(annIds)))
        ref_mask = self._get_mask(ref_mask, choice)
        query_mask = self._get_mask(query_mask, choice)
        nobjects = int(torch.max(ref_mask))
            
        # if self.is_train:
        #     ref_mask, query_mask = self._filter([ref_mask, query_mask])

        if nobjects == 0:
            return self.__getitem__(random.randrange(0, len(self.frame_list)))
        
        return (ref_img, ref_mask, query_img, nobjects), (query_mask,)

    def __len__(self):
        return len(self.frame_list)

class DAVISTripletDataset(DAVISCoreDataset):
    def __init__(self, max_npairs=-1, **kwargs):
        super().__init__(**kwargs)

        self.max_npairs = max_npairs

        # Generate frames
        self.frame_list = []
        for video_name in self.video_names:
            png_pair = self.get_frame(self.mode, video_name)
            nobjects = self.infos[video_name]['nobjects']
            for pair in png_pair:
                support_anno = video_name + "/" + pair[0]
                pres_anno = video_name + "/" + pair[1]
                query_anno = video_name + "/" + pair[2]
                self.frame_list.append(
                    (support_anno, pres_anno, query_anno, nobjects))

    def get_frame(self, mode, video_name):
        images = sorted(os.listdir(str(self.annotation_path / video_name)))
        n = len(images)
        min_skip = self.min_skip
        max_skip = min(n - 1,
                       self.max_skip if self.max_skip != -1 else n - 1)

        if mode == 0:
            return [(images[i], images[j], images[k])
                    for i in range(0, n)
                    for j in range(i+1, n)
                    for k in range(j+1, n)
                    if min_skip <= j - i <= max_skip and min_skip <= k - j <= max_skip]
        elif mode == 1:
            return [(images[0], images[k - 1], images[k])
                    for k in range(1 + min_skip, max_skip + 1)]
        elif mode == 2:
            indices = [(i, j, k)
                       for i in range(n-2)
                       for j in range(i+1, n-1)
                       for k in range(j+1, n)
                       if min_skip <= j - i <= max_skip and min_skip <= k - j <= max_skip]
            max_npairs = min(len(indices),
                             self.max_npairs if self.max_npairs != -1 else len(indices))
            indices = random.sample(indices, k=max_npairs)
            return [(images[i], images[j], images[k]) for i, j, k in indices]
        elif mode == 4:
            return [(images[0], images[i], images[j])
                    for i in range(1, n)
                    for j in range(i, n)
                    if min_skip <= i <= max_skip and min_skip <= j - i <= max_skip]
        else:
            raise Exception('Unknown mode')

    def _get_mask(self, mask, anns):
        new_mask = np.zeros(mask.shape, dtype=np.int8)
        for i, ann in enumerate(anns):
            new_mask[mask == ann] = i + 1
        new_mask = torch.LongTensor(np.array(new_mask))

        return new_mask

    def __getitem__(self, inx):
        support_anno_name, pres_anno_name, query_anno_name, nobjects = self.frame_list[inx]

        ref_img, ref_mask = self._load_frame(support_anno_name,
                                             self._augmentation)
        inter_img, inter_mask = self._load_frame(pres_anno_name,
                                                 self._augmentation)
        query_img, query_mask = self._load_frame(query_anno_name,
                                                 self._augmentation)

        if self.is_train:
            ref_mask, inter_mask, query_mask = self._filter(
                [ref_mask, inter_mask, query_mask])

        mask = np.array(ref_mask)
        annIds = list(set(list(mask.reshape(-1))))[1:]
        max_annot = 3
        choice = random.sample(annIds, k = min(max_annot, len(annIds)))
        ref_mask = self._get_mask(ref_mask, choice)
        inter_mask = self._get_mask(inter_mask, choice)
        query_mask = self._get_mask(query_mask, choice)
        nobjects = int(torch.max(ref_mask))
            
        if self.is_train:
            ref_mask, query_mask = self._filter([ref_mask, query_mask])

        if nobjects == 0:
            return self.__getitem__(random.randrange(0, len(self.frame_list)))

        return (ref_img, ref_mask, inter_img, query_img, nobjects), (inter_mask, query_mask)

    def __len__(self):
        return len(self.frame_list)