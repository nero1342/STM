from pycocotools.coco import COCO 
from PIL import Image 
import numpy as np 
from torch.utils import data 
from pathlib import Path 
import os 

import random 

class COCODataset(data.Dataset):
    def __init__(self, 
        img_folder=None,
        annot_file=None,
        max_annot = 3
    ):
        super(COCODataset, self).__init__() 

        assert img_folder is not None, "Missing image folder path!"
        assert annot_file is not None, "Missing annotations json file, should be a coco format json file!"

        self.img_folder = Path(img_folder)
        self.coco = COCO(annot_file)
        self.imgIds = sorted(self.coco.getImgIds()) 

        self.max_annot = max_annot

        if not os.path.isfile("palette.png"):
            print("Downloading palette...")
            os.system("gdown --id 1DT0b0WeGxiLQVcUai4hR3KIWwyh56rKu -O palette.png")
        self.palette = Image.open("palette.png").getpalette()
        #print(self.palette, Image.open("00000.png").size)

    def __len__(self): 
        return len(self.imgIds)
        
    def __getitem__(self, index):
        imgId = self.coco.loadImgs(self.imgIds[index])[0]

        img_path = self.img_folder / imgId['file_name']
        img = Image.open(img_path).convert('RGB')

        # Get annotations of this image
        annIds = self.coco.getAnnIds(imgId['id']) 
        
        # Choose randomly x annot in annIds
        annIds = random.sample(annIds, k = min(self.max_annot, len(annIds)))

        anns = self.coco.loadAnns(annIds) 
        mask = self._get_mask(imgId['height'], imgId['width'], anns)
        
        mask = Image.fromarray(mask).convert('P')
        if self.palette:
            mask.putpalette(self.palette)
        
        return img, mask
    
    def _get_mask(self, h, w, anns):
        combined_mask = np.zeros((h, w), dtype=np.int8)
        ids = np.random.choice(np.arange(10), size=len(anns))
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)
            combined_mask[mask == 1] = ids[i] + 1
        return combined_mask