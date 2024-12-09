import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import sys
sys.path.append(os.getcwd())

import kiui
from core.options_pm import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

from PIL import Image, ImageDraw
import json

import open3d as o3d

import tyro
from core.options import AllConfigs

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class ObjaverseHQDataset(Dataset):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        self.num_drag_samples = 10
        self.num_views = 12
        self.num_max_drags = 5
        self.num_frames = 5

        action_dirs = []

        filelist = self.opt.val_filelist
        with open(filelist, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                action_dirs.append(line)

        self.image_paths = []
        self.deform_image_paths = []
        self.cur_frame_ids = []
        self.deform_frame_ids = []
        self.zero123_items = []
        zero123_filelist = self.opt.zero123_val_filelist

        with open(zero123_filelist, 'r') as f:
            zero123_lines = [line.strip() for line in f.readlines()]

        for i, action_dir in enumerate(action_dirs):
            if os.path.isdir(action_dir):
                valid_image_list = self.get_valid_image_list(action_dir)
                if len(valid_image_list) != 5:
                    continue
                if not self.get_num_views(action_dir):
                    continue
                if not self.judge_action(action_dir):
                    continue
                for cur_frame_id, image in enumerate(valid_image_list[:1]):
                    other_action_images = self.get_other_action_ids(action_dir, image, valid_image_list)
                    for deform_frame_id, other_action_image in enumerate(other_action_images):
                        self.image_paths.append(os.path.join(action_dir, image))
                        self.zero123_items.append(zero123_lines[i])

                        deform_image_path = os.path.join(action_dir, other_action_image)
                        self.deform_image_paths.append(deform_image_path)

                        self.cur_frame_ids.append(cur_frame_id)
                        if deform_frame_id >= cur_frame_id:
                            deform_frame_id += 1
                        self.deform_frame_ids.append(deform_frame_id)

        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        print(f"Dataset size: {len(self.image_paths)}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def get_valid_image_list(self, path):
        images = []
        for file in os.listdir(path):
            if file.startswith('000') and file.endswith('.png'):
                images.append(file)
        
        images = sorted(images, key=lambda x: int(x.split('.')[0].split('_')[1]))

        return images[:self.num_frames]

    def get_num_views(self, path):
        image = None
        for file in os.listdir(path):
            if file.startswith('000') and file.endswith('.png'):
                image = file
                break
        
        for i in range(12):
            new_file = f'{i:03d}{image[3:]}'
            if not os.path.exists(os.path.join(path, new_file)):
                return False
            else:
                return True


    def get_other_action_ids(self, image_dir, image_name, valid_image_list):
        results = []
        for image in os.listdir(image_dir):
            if image != image_name and image in valid_image_list:
                results.append(image)

        results.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
        return results
    
    def judge_action(self, image_dir):
        for i in range(self.num_drag_samples):
            if not os.path.exists(os.path.join(image_dir, f'sample_{i:03d}.json')):
                return False

            with open(os.path.join(image_dir, f'sample_{i:03d}.json'), 'r') as f:
                try:
                    sample = json.load(f)
                except json.decoder.JSONDecodeError:
                    return False
            
            drags = sample["all_tracks"][0]
            if len(drags) == 0:
                return False
        
        return True

    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians

    def process_state_data(self, image_path0, zero123_uid, camera_dir, vids, is_input=False):
        images = []
        masks = []
        cam_poses = []
        camera_matrixes = []

        image_name = image_path0.split('/')[-1]

        for vid_cnt, vid in enumerate(vids):
            image_name_vid = os.path.join(os.path.dirname(image_path0), f"{vid:03d}" + image_name[3:])
            camera_path_vid = os.path.join(camera_dir, f"{vid:03d}.npy")

            if is_input and vid_cnt > 4 and vid_cnt < 4:
                image_name_vid= os.path.join(zero123_uid, f'{vid:03d}_rgba.png')

            with open(image_name_vid, 'rb') as f:
                image = np.frombuffer(f.read(), np.uint8)
            meta = np.load(camera_path_vid)

            image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
            
            camera_matrix = torch.eye(4)
            camera_matrix[:3, 0] = torch.from_numpy(meta[0, :3])
            camera_matrix[:3, 1] = -torch.from_numpy(meta[1, :3])
            camera_matrix[:3, 2] = -torch.from_numpy(meta[2, :3])
            camera_matrix[:3, 3] = torch.from_numpy(np.linalg.inv(-meta[:3, :3]) @ meta[:3, 3])

            c2w = camera_matrix
            c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]

            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg

            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)
            camera_matrixes.append(camera_matrix)


        images = torch.stack(images, dim=0) # [V, 3, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        camera_matrixes = torch.stack(camera_matrixes, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        return images, masks, cam_poses, camera_matrixes


    def __getitem__(self, idx):
        image_path0 = self.image_paths[idx]
        image_path0_output = self.deform_image_paths[idx]
        zero123_name = self.zero123_items[idx]

        results = {}

        vids = np.arange(0, 12).tolist()
    
        camera_dir = os.path.dirname(os.path.dirname(image_path0))
        camera_dir_output = os.path.dirname(os.path.dirname(image_path0_output))

        images, masks, cam_poses, camera_matrixes = self.process_state_data(image_path0, zero123_name, camera_dir, vids, is_input=True)
        images_output, masks_output, cam_poses_output, camera_matrix_output = self.process_state_data(image_path0_output, zero123_name, camera_dir_output, vids, is_input=False)
        #######################################################

        results['images_input'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images_output, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks_output.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos
        
        # Obtain Drag
        drag_roots = os.path.dirname(image_path0)
        sample_id = np.random.randint(self.num_drag_samples)

        for sample_id in range(self.num_drag_samples):
            sample_fpath = os.path.join(drag_roots, f"sample_{sample_id:03d}.json")

            with open(sample_fpath, 'r') as f:
                sample = json.load(f)
            assert len(sample["all_tracks"]) == self.num_views
            drags = sample["all_tracks"][0]

            if len(drags) > self.num_max_drags:
                drags = random.sample(drags, self.num_max_drags)

            drags = torch.tensor(drags).permute(1,0,2).float()  # num_frames, num_points, 2

            cur_frame_id = self.cur_frame_ids[idx]
            deform_frame_id = self.deform_frame_ids[idx]

            drags_start = drags[cur_frame_id,:,:].clone()
            drags_end = drags[deform_frame_id,:,:].clone()

            final_drags_start = []
            final_drags_end = []

            for drag_id in range(drags_start.shape[0]):
                if torch.norm((drags_start[drag_id] - drags_end[drag_id]).to(torch.float32)) >= 5:
                    final_drags_start.append(drags_start[drag_id])
                    final_drags_end.append(drags_end[drag_id])
        
            if len(final_drags_start) != 0:
                final_drags_start = torch.stack(final_drags_start, dim=0)
                final_drags_end = torch.stack(final_drags_end, dim=0)
                break
            else:
                final_drags_start = torch.zeros_like(drags_start)
                final_drags_end = torch.zeros_like(drags_end)

        drags_start = final_drags_start.clone()
        drags_end = final_drags_end.clone()

        drags_start = drags_start.unsqueeze(0).repeat(self.opt.num_input_views,1,1)
        drags_end = drags_end.unsqueeze(0).repeat(self.opt.num_input_views,1,1)

        drags_start = torch.clamp(drags_start, 0, 511)
        drags_end  = torch.clamp(drags_end, 0, 511)

        if drags_start.shape[1] < self.num_max_drags:
            drags_start = torch.cat([drags_start, torch.zeros(4, self.num_max_drags - drags_start.shape[1], 2)], dim=1)
            drags_end = torch.cat([drags_end, torch.zeros(4, self.num_max_drags - drags_end.shape[1], 2)], dim=1)
        
        results['drags_start'] = drags_start
        results['drags_end'] = drags_end

        return results