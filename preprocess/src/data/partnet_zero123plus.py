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


class PartNetData(Dataset):
    def __init__(self,
        filelist='/data/chenjh/gaomx/workspace/blenderRender/LGM/filelist/filelist_multistates_val.txt',
        validation=False,
    ):
        self.paths = []
        with open(filelist, 'r') as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                self.paths.append(line.strip())
        
        
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
        while True:
            image_path = self.paths[index]

            '''background color, default: white'''
            bkg_color = [1., 1., 1.]

            img_list = []
            try:
                for idx in range(7):
                    img, alpha = self.load_im(os.path.join(image_path, '%03d.png' % idx), bkg_color)
                    img_list.append(img)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break

        imgs = torch.stack(img_list, dim=0).float()

        data = {
            'cond_imgs': imgs[0],           # (3, H, W)
            'target_imgs': imgs[1:],        # (6, 3, H, W)
        }
        return data