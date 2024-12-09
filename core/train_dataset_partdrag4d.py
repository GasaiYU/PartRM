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
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

from PIL import Image
import json

import open3d as o3d
from PIL import Image, ImageDraw

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class PartDrag4DDatset(Dataset):
    def __init__(self, opt: Options):
        self.opt = opt  

        self.items = []
        self.statej_items = []
        self.gs_statej_paths = []
        self.gs_statei_paths = []

        filelist = opt.train_filelist
        
        with open(filelist, 'r') as f:
            for i, file in enumerate(f.readlines(), start=1):
                file = file.strip()
                if file.startswith('#'):
                    continue
                # Get the state id
                state = int(file.strip().split('/')[-1].split('_')[2])
                for j in range(0, 6):
                    if state != j:
                        base, foldername = file.strip().rsplit('/', 1)
                        parts = foldername.split('_')
                        statei_name = '_'.join(parts)
                        parts[2] = str(j)
                        statej_name = '_'.join(parts)
                        foldername_statej = os.path.join(base, statej_name)
                        
                        if not os.path.exists(foldername_statej):
                            continue
                        
                        # Read gaussian path
                        gs_statej_path = os.path.join(opt.base_file_path, f'eval_gaussians_{statej_name}.ply')
                        if not os.path.exists(gs_statej_path):
                            continue

                        gs_statei_path = os.path.join(opt.base_file_path, f'eval_gaussians_{statei_name}.ply')
                        if not os.path.exists(gs_statej_path):
                            continue

                        item_path = file.strip()
                        statej_item_path = foldername_statej
                        
                        self.items.append(item_path)
                        self.statej_items.append(statej_item_path)
                        self.gs_statej_paths.append(gs_statej_path)
                        self.gs_statei_paths.append(gs_statei_path)
        
        print('self.items.__len__()=',self.items.__len__()) 
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1
        
    def __len__(self):
        return len(self.items)
    
    def process_state_data(self, uid, vids):
        images = []
        masks = []
        cam_poses = []
        camera_matrices = []
        vid_cnt = 0

        for vid in vids:
            try:
                image_path = os.path.join(uid, f'{vid:03d}.png')
                camera_path = os.path.join(uid, f'{vid:03d}_camera.json')
                
                with open(image_path, 'rb') as f:
                    image = np.frombuffer(f.read(), np.uint8)
                    
                with open(camera_path, 'r') as f:
                    meta = json.load(f)

                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                
                camera_matrix = torch.eye(4)
                camera_matrix[:3, 0] = torch.tensor(meta["x"])
                camera_matrix[:3, 1] = -torch.tensor(meta["y"])
                camera_matrix[:3, 2] = -torch.tensor(meta["z"])
                camera_matrix[:3, 3] = torch.tensor(meta["origin"])
                c2w = camera_matrix
                c2w = c2w.clone().float().reshape(4, 4)

            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
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
            camera_matrices.append(camera_matrix)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            print(len(images))
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, 3, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        camera_matrices = torch.stack(camera_matrices, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        
        return images, masks, cam_poses, camera_matrices
    
    
    def ndc2Pix(self, v, S):
        return ((v + 1.0) * S - 1.0) * 0.5
    
    def local2world(self, drags, scale, world_matrix):
        drags = drags.clone().detach()
        scale = torch.tensor(scale, dtype=torch.float32).clone().detach()  
        translation = torch.tensor(world_matrix, dtype=torch.float32)
        
        scaled_points = drags * scale
        world_points = scaled_points + translation
        
        return world_points
    
    def project_drags(self, drags, cam_view):
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

    def __getitem__(self, idx):
        uid = self.items[idx]
        uid_output = self.statej_items[idx]
        parts_out = uid_output.split('/')[-1].split('_') 
        parts_in = uid.split('/')[-1].split('_')

        results = {} 
        
        render_id = uid.split('/')[-1]
        mesh_id = render_id.split('_')[0]
        
        # Find the class
        class_names = [name for name in os.listdir(self.opt.mesh_base) if os.path.isdir(os.path.join(self.opt.mesh_base, name))]
        cur_class = None
        for classname in class_names:
            item_base = os.path.join(self.opt.mesh_base, classname, mesh_id)
            if os.path.exists(item_base):
                cur_class = classname
                break
        if cur_class is None:
            raise Exception(f"{uid} drag base not found") 
        
        statei_id = '_'.join(parts_in[1:])
        statej_id = '_'.join(parts_out[1:])
            
        # Get Motion idx
        pcd0 = np.asarray(o3d.io.read_point_cloud(os.path.join(self.opt.mesh_base, cur_class, parts_out[0], 'motion', f"{statei_id}.ply")).points)
        pcd1 = np.asarray(o3d.io.read_point_cloud(os.path.join(self.opt.mesh_base, cur_class, parts_out[0], 'motion', f"{statej_id}.ply")).points)
        pcd_idx = np.where((pcd0 - pcd1).sum(1) != 0)[0]
        if pcd_idx.shape[0] < self.opt.num_drags:
            pcd_idx = np.random.choice(pcd_idx, self.opt.num_drags)
        
        # Get visible idx
        if os.path.exists(os.path.join(self.opt.mesh_base, cur_class, parts_out[0], 'motion', f"{statei_id}_cam0_visible.npy")):
            visible_idx = np.load(os.path.join(self.opt.mesh_base, cur_class, parts_out[0], 'motion', f"{statei_id}_cam0_visible.npy"))
            inter_idx = np.intersect1d(visible_idx, pcd_idx)
            rand_idx = np.random.permutation(inter_idx)[:self.opt.num_drags]
        else:
            rand_idx = np.array([])
        
        # If the visible points are so sparse, we use all the moving points between 2 pcds.
        if rand_idx.shape[0] > int(self.opt.num_drags * 2 // 3):
            if rand_idx.shape[0] < self.opt.num_drags:
                rand_idx = np.append(rand_idx, np.random.choice(inter_idx, self.opt.num_drags - rand_idx.shape[0]))
            
            results['drags_start'] = torch.tensor(pcd0[rand_idx, :], dtype=torch.float32)
            results['drags_end'] = torch.tensor(pcd1[rand_idx, :], dtype=torch.float32)
        else:       
            start_action = pcd0[pcd_idx]
            end_action = pcd1[pcd_idx]
            
            rand_idx = np.random.permutation(start_action.shape[0])[:self.opt.num_drags]
            results['drags_start'] = torch.tensor(start_action[rand_idx, :], dtype=torch.float32)
            results['drags_end'] = torch.tensor(end_action[rand_idx, :], dtype=torch.float32)
            
        # load num_views images
        images = []
        cam_poses = []
        
        images_output = []
        masks_output = []
        cam_poses_output = []

        vids = np.arange(0, 12)
            
        images, masks, cam_poses, camera_matrix = self.process_state_data(uid, vids)
        images_output, masks_output, cam_poses_output, camera_matrix_output = self.process_state_data(uid_output, vids)
        results['images_input'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        
        # apply random grid distortion to simulate 3D inconsistency
        if random.random() < self.opt.prob_grid_distortion:
            images_input[1:] = grid_distortion(images_input[1:])
        # apply camera jittering (only to input!)
        if random.random() < self.opt.prob_cam_jitter:
            cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

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

        # changed cam_poses to output camera poses
        # opengl to colmap camera for gaussian renderer
        cam_poses_output[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses_output).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses_output[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos
        
        # project 3d drags to 2d        
        camera_matrix1 = torch.stack([camera_matrix[0], camera_matrix[0], camera_matrix[0], camera_matrix[0]], dim=0)
        results['drags_start'] = torch.clamp(self.project_drags(results['drags_start'], camera_matrix1), 0, 511)
        results['drags_end'] = torch.clamp(self.project_drags(results['drags_end'], camera_matrix1), 0, 511)
        
        if self.opt.perturb_drags and random.random() < 0.5:
            perturb = torch.randn_like(results['drags_end']) * 5
            results['drags_end'] = torch.clamp(results['drags_end'] + perturb, 0, 511)

        # Load state5 gaussian  
        results['gt_gaussians'] = self.load_ply(self.gs_statej_paths[idx])
        results['uid'] = uid.split('/')[-1]
        results['uid_output'] = uid_output.split('/')[-1]   

        return results
