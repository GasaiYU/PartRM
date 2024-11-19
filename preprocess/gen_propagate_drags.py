# This is only for PartDrag4D dataset

import torch
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F

import os
import cv2

import json
import random
import open3d as o3d
import copy

from tqdm import tqdm

from PIL import Image, ImageDraw

from segment_anything import SamPredictor, sam_model_registry
import argparse

def gen_rand_num(num_range, cur_num):
    while True:
        a = random.randint(0, num_range - 1)
        if a != cur_num:
            return a

def sam_predict(predictor: SamPredictor, image, query_point):
    predictor.set_image(image=image)
    labels = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=query_point,
        point_labels=labels,
        multimask_output=True
    )

    min_sum = 10000000
    for mask in masks:
        if mask.sum() < min_sum:
            selected_mask = mask
            min_sum = mask.sum()
            
    return selected_mask

class DragPropagateDataset(Dataset):
    def __init__(self, filelist, pcd_base, render_base):
        self.pcd_base = pcd_base
        self.render_base = render_base
        
        self.images, self.cameras = [], []
        
        with open(filelist, 'r') as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue

                render_dir = line.strip()

                self.images.append(os.path.join(render_dir, f'000.png'))
                self.cameras.append(os.path.join(render_dir, f'000_camera.json'))
        
        self.class_names = [name for name in os.listdir(pcd_base) if os.path.isdir(os.path.join(pcd_base, name))]
        self.cam_radius = 2.4
        self.h = self.w = 512
        self.fov = 49.1
        
    def __len__(self):
        return len(self.images)
    
    
    def ndc2Pix(self, v, S):
        return ((v + 1.0) * S - 1.0) * 0.5
    
    
    def local2world(self, drags, scale, world_matrix):
        drags = drags.clone().detach()
        scale = torch.tensor(scale, dtype=torch.float32).clone().detach()  
        translation = torch.tensor(world_matrix, dtype=torch.float32)
        
        scaled_points = drags * scale
        world_points = scaled_points + translation
        
        return world_points
    
    def project_drag(self, drags, cam_view):
        # local2world transformation using blender scale and world matrix
        scale =  0.42995866893870127
        world_matrix = (-0.0121, -0.0070, 0.0120)
        drags = self.local2world(drags, scale, world_matrix)
        
        # use blender fixed projection matrix
        proj_matrix = [
            [2.777777671813965, 0.0000,  0.0000,  0.0000],
            [0.0000, 2.777777671813965,  0.0000,  0.0000],
            [0.0000, 0.0000, -1.0001999139785767, -0.20002000033855438],
            [0.0000, 0.0000, -1.0000,  0.0000]
        ]
        cam_proj = torch.tensor(proj_matrix, dtype=torch.float32)
        
        ndc_coords_2d_list = []

        for i in range(cam_view.size(0)):
            for j in range(drags.size(0)):
                view_matrix = cam_view[i]

                view_matrix = torch.inverse(view_matrix)
                proj_matrix = cam_proj
                
                point_3D = drags[j]
                point_3D_homogeneous = torch.cat([point_3D, torch.tensor([1.0])], dim=0)

                camera_coords = torch.matmul(view_matrix, point_3D_homogeneous)
                clip_coords = torch.matmul(proj_matrix, camera_coords)

                ndc_coords = clip_coords[:3] / clip_coords[3]
                ndc_coords_2d_list.append(ndc_coords[:2].tolist())

        ndc_coords_2d_tensor = torch.tensor(ndc_coords_2d_list).view(cam_view.size(0), drags.size(0), 2)
        
        # drag ndc to pix
        S = torch.tensor([512, 512])
        drags_2d = S - self.ndc2Pix(ndc_coords_2d_tensor, S)
        
        return drags_2d
    
    
    def __getitem__(self, idx):
        # ----- Load image
        image_path = self.images[idx]
        with open(image_path, 'rb') as f:
            image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        image = torch.from_numpy(image).float() / 255.0
        
        image = image.permute(2, 0, 1) # [4, 512, 512]
        mask = image[3:4] # [1, 512, 512]
        image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
        image = image[[2,1,0]].contiguous() # bgr to rgb
        
        # ----- Load camera parameters
        camera_path = self.cameras[idx]
        with open(camera_path, 'r') as f:
            meta = json.load(f)
        
        camera_matrix = torch.eye(4)
        camera_matrix[:3, 0] = torch.tensor(meta["x"])
        camera_matrix[:3, 1] = -torch.tensor(meta["y"])
        camera_matrix[:3, 2] = -torch.tensor(meta["z"])
        camera_matrix[:3, 3] = torch.tensor(meta["origin"])
        c2w = camera_matrix
        c2w = c2w.clone().float().reshape(4, 4)   
        
        # ----- Get 3D Drags
        mesh_id = image_path.split('/')[-2].split('_')[0]
        cur_class = None
        for class_name in self.class_names:
            item_base = os.path.join(self.pcd_base, class_name, mesh_id)
            if os.path.exists(item_base):
                cur_class = class_name
                break
        if cur_class is None:
            raise Exception(f"{mesh_id} drag base not found")
        
        # ----- Get 2D Drag
        pcd_path = os.path.join(self.pcd_base, class_name, mesh_id, 'motion')
        cur_parts = image_path.split('/')[-2].split('_')
        rand_parts = copy.deepcopy(cur_parts)
        
        cur_motion_id = int(cur_parts[2])
        rand_motion_id = gen_rand_num(6, cur_motion_id)
        rand_parts[2] = str(rand_motion_id)
        cur_pcd_name = '_'.join(cur_parts[1:])
        rand_pcd_name = '_'.join(rand_parts[1:])
        
        pcd0 = np.asarray(o3d.io.read_point_cloud(os.path.join(pcd_path, f'{cur_pcd_name}.ply')).points)
        pcd1 = np.asarray(o3d.io.read_point_cloud(os.path.join(pcd_path, f'{rand_pcd_name}.ply')).points)
        pcd_rand_idx = np.where((pcd0 - pcd1).sum(1) != 0)[0]

        surface_2d_index = np.load(os.path.join(pcd_path, f'{cur_pcd_name}_visible.npy'))
        surface_2d_index = np.intersect1d(surface_2d_index, pcd_rand_idx)

        if surface_2d_index.shape[0] > 0:
            rand_2d_index = random.choices(surface_2d_index, k=1)
        else:
            rand_2d_index = random.choices(pcd_rand_idx, k=1)

        rand_2d_drag_3d_start = torch.from_numpy(pcd0[rand_2d_index]).float()
        rand_2d_drag_start = self.project_drag(rand_2d_drag_3d_start, camera_matrix.unsqueeze(0))
        rand_2d_drag_start = torch.clamp(rand_2d_drag_start, 0, 511)
        
        return rand_2d_drag_start, image_path
    
def main(val_filelist, mesh_base, render_base, save_dir, sample_num=10):
    drag_propagate_dataset = DragPropagateDataset(filelist=val_filelist, pcd_base=mesh_base, render_base=render_base)
    sam = sam_model_registry['vit_h'](checkpoint='sam_ckpt/sam_vit_h_4b8939.pth')
    sam.to('cuda')
    predictor = SamPredictor(sam)

    for i in tqdm(range(len(drag_propagate_dataset))):
        data, image_path = drag_propagate_dataset[i]
        query_point = np.array([[int(data[0, 0, 0].item()), int(data[0, 0, 1].item())]])
        sam_image= cv2.imread(image_path)
        sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
        mask = sam_predict(predictor, sam_image, query_point)

        true_indices = np.argwhere(mask)
        sampled_indices = true_indices[np.random.permutation(sampled_indices.shape[0])[:sample_num]]
        render_id = image_path.split('/')[-2]
        os.makedirs(os.path.join(save_dir, render_id), exist_ok=True)
        np.save(os.path.join(save_dir, render_id, 'propagated_indices.npy'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_filelist', default='../filelist/val_filelist.txt', help='The path of the val filelist')
    parser.add_argument('--mesh_base', default='../PartDrag4D/data/processed_data_partdrag4d', help='The dir of the processed data which contains the mesh')
    parser.add_argument('--render_base', default='../PartDrag4D/data/render_PartDrag4D', help='The dir of the rendered images')
    parser.add_argument('--sample_num', default=10, help="The number of the sample points")
    parser.add_argument('--save_dir', default='./propagated_drags', help="The path to save the propagated drags.")
    args = parser.parse_args()

    main(val_filelist=args.val_filelist, mesh_base=args.mesh_base, render_base=args.render_base, 
         save_dir=args.save_dir, sample_num=args.sample_num)