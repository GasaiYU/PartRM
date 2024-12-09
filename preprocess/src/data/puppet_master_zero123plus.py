import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path

from src.utils.train_util import instantiate_from_config

PUPPET_MASTER_BASE = '/gpfs/essfs/iat/Tsinghua/gaomx/gaomx/data/puppet_master_fixedphi/'

class ObjaverseData(Dataset):
    def __init__(self,
        root_dir=PUPPET_MASTER_BASE,
        validation=False,
    ):
        self.root_dir = Path(root_dir)

        self.paths = []

        for action_dir in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(root_dir, action_dir)):
                for drag_dir in os.listdir(os.path.join(root_dir, action_dir)):
                    if os.path.isdir(os.path.join(root_dir, action_dir, drag_dir)):
                        for image in os.listdir(os.path.join(root_dir, action_dir, drag_dir)):
                            if image.endswith('.png') and image.startswith('000'):
                                self.paths.append(os.path.join(root_dir, action_dir, drag_dir, image))

        if validation:
            self.paths = self.paths[-16:] # used last 16 as validation
        else:
            self.paths = self.paths[:-16]
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def __getitem__(self, index):
        image_path = self.paths[index]

        '''background color, default: white'''
        bkg_color = [1., 1., 1.]

        img_list = []
        for idx in range(7):
            image_name = os.path.basename(image_path)
            load_image_name = f"{idx:03d}{image_name[3:]}"
            load_image_path = os.path.join(os.path.dirname(image_path), load_image_name)
            img, alpha = self.load_im(load_image_path, bkg_color)
            img_list.append(img)
        
        imgs = torch.stack(img_list, dim=0).float()

        data = {
            'cond_imgs': imgs[0],           # (3, H, W)
            'target_imgs': imgs[1:],        # (6, 3, H, W)
        }
        return data
