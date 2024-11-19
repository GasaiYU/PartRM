# Copyright (c) Facebook, Inc. and its affiliates.import math
import os
import torch
import numpy as np
import glob
import natsort
from torch.autograd import Variable
import shutil

import math

import json
import csv
import open3d as o3d
from process_data_textured_uv import load_obj_with_texture, save_combined_obj_with_texture
import random

device = torch.device("cuda:0")
torch.cuda.set_device(device)

cad_folder = '../data/processed_data_partdrag4d'
cad_classes = ['Dishwasher', 'Laptop', 'Microwave', 'Oven', 
               'Refrigerator', 'StorageFurniture', 'WashingMachine', 'TrashCan']


# helper function for computing roation matrix in 3D
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = torch.empty(3,3)

    rot_mat[0,0] = aa + bb - cc - dd
    rot_mat[0,1] = 2 * (bc + ad)
    rot_mat[0,2] = 2 * (bd - ac)

    rot_mat[1,0] = 2 * (bc - ad)
    rot_mat[1,1] = aa + cc - bb - dd
    rot_mat[1,2] = 2 * (cd + ab)

    rot_mat[2,0] = 2 * (bd + ac)
    rot_mat[2,1] = 2 * (cd - ab)
    rot_mat[2,2] = aa + dd - bb - cc

    return rot_mat

# helper function for loading and merging meshes
def merge_meshes(obj_path):
    
    material_dict = {}
    mtl_files = []
    
    verts_list = torch.empty(0,3)
    vert_normals_list = np.empty((0,3))
    vert_textures_list = np.empty((0,2))
    
    faces_list = torch.empty(0,3).long()
    faces_vt_list = []
    faces_vn_list = []
    num_vtx = [0]

    # merge meshes, load in ascending order
    meshes = natsort.natsorted(glob.glob(obj_path+'/parts_ply/*.obj'))
    for part_mesh in meshes:
        print('loading %s' %part_mesh)
        
        verts, vertex_normals, vertex_textures, faces, mtl_file, material_ranges = load_obj_with_texture(part_mesh)
        mtl_files.extend(mtl_file)
        
        verts = torch.from_numpy(np.asarray(verts)).float()
        vertex_normals = np.asarray(vertex_normals)
        vertex_textures = np.asarray(vertex_textures)
        
        # reindex material dict
        for _, face_range in material_ranges.items():
            face_range['start'] += faces_list.shape[0]
            face_range['end'] += faces_list.shape[0]
        material_dict.update(material_ranges)
        
        # reindex face coordinates
        faces_v = torch.from_numpy(np.asarray([[val + verts_list.shape[0] for val in sublist] for sublist in faces['v']])).long()
        faces_vt = [[val if val is None else val + vert_textures_list.shape[0]
                for val in sublist] for sublist in faces['vt']]
        faces_vn = [[val if val is None else val + vert_normals_list.shape[0]
                        for val in sublist] for sublist in faces['vn']]
        
        # add to list
        verts_list = torch.cat([verts_list, verts])
        faces_list = torch.cat([faces_list, faces_v])
        vert_normals_list = np.concatenate([vert_normals_list, vertex_normals])
        if vertex_textures.shape[0] != 0:
            vert_textures_list = np.concatenate([vert_textures_list, vertex_textures])
        faces_vt_list.extend(faces_vt)
        faces_vn_list.extend(faces_vn)
        
        num_vtx.append(verts_list.shape[0])

    verts_list = verts_list.to(device)
    faces_list = faces_list.to(device)

    return verts_list, vert_normals_list, vert_textures_list, faces_list, faces_vt_list, faces_vn_list, num_vtx, mtl_files, material_dict


def find_indices(tensor, numbers_list):
    indices = []
    for number in numbers_list:
        mask = (tensor == number)
        index_tuples = torch.nonzero(mask, as_tuple=True)
        indices.extend(index_tuples[0].tolist())
    return torch.tensor(indices).long().to(device)

def deform_part(idx, key, origin_verts, pts, deform_num):
    
    deformed_data = {}
    
    jointData = motion[key]

    if 'type' not in jointData:
        return deformed_data
    
    # rotation part
    if jointData and jointData['type'] == 'revolute':
        part_div = jointData['part_div']
        
        start = num_vtx[idx]
        end = num_vtx[idx+1]

        rot_orig = torch.FloatTensor(jointData['axis']['origin']).to(device)
        if not None in jointData['axis']['direction']:
            rot_axis = torch.FloatTensor(jointData['axis']['direction']).to(device)
        else:
            return deformed_data

        aa = math.pi*jointData['limit']['a'] / 180.0
        bb = math.pi*jointData['limit']['b'] / 180.0

        angles = np.linspace(aa, bb, num=deform_num)

        for k, angle in enumerate(angles):

            verts = origin_verts.clone()
            faces = faces_list.clone()
            all_pts_clone = pts.clone()
            
            verts[start:end, 0] -= rot_orig[0]
            verts[start:end, 1] -= rot_orig[1]
            verts[start:end, 2] -= rot_orig[2]

            # Deal with Point Cloud
            pcd_idx = find_indices(all_labels, part_div)
            
            start_cor = all_pts_clone[pcd_idx,:].clone()
            all_pts_clone[pcd_idx] -= rot_orig
            
            # rotate around local axis [-1 0 0]
            init_value = torch.tensor([angle])
            theta = Variable(init_value.cuda())
            rot_mat = rotation_matrix(rot_axis, theta).float()  # 3x3

            # Get the move coordinates
            verts[start:end,:] = torch.t(torch.mm(rot_mat.to(device),
                                            torch.t(verts[start:end,:])))

            # local coordinate to world coordinate
            verts[start:end, 0] += rot_orig[0]
            verts[start:end, 1] += rot_orig[1]
            verts[start:end, 2] += rot_orig[2]

            # Deal with Point Cloud
            all_pts_clone[pcd_idx] = torch.t(torch.mm(rot_mat.to(device),
                                            torch.t(all_pts_clone[pcd_idx]))).float()
            all_pts_clone[pcd_idx] += rot_orig
            
            end_cor = all_pts_clone[pcd_idx, :].clone()
            
            deformed_data[key+'_'+str(k)] = {
                'verts': verts,
                'faces': faces,
                'all_pts_clone': all_pts_clone,
                'pcd_idx': pcd_idx,
                'start_cor': start_cor,
                'end_cor': end_cor
            }
    
    # translation part
    elif jointData and jointData['type'] == 'prismatic':
        part_div = jointData['part_div']
        start = num_vtx[idx]
        end = num_vtx[idx+1]

        trans_orig = torch.FloatTensor(jointData['axis']['origin']).to(device)
        trans_axis = torch.FloatTensor(jointData['axis']['direction']).to(device)

        aa = jointData['limit']['a']
        bb = jointData['limit']['b']

        trans_lens = np.linspace(aa, bb, num=deform_num)

        for k, tran_len in enumerate(trans_lens):

            verts = origin_verts.clone()
            faces = faces_list.clone()
            all_pts_clone = pts.clone()

            # world coordinate to local coordinate (rotation origin)
            # rand_index = torch.randint(start, end, (10,))

            verts[start:end, 0] -= trans_orig[0]
            verts[start:end, 1] -= trans_orig[1]
            verts[start:end, 2] -= trans_orig[2]
            
            # Deal with Point Cloud
            pcd_idx = find_indices(all_labels, part_div)

            start_cor = all_pts_clone[pcd_idx,:].clone()
            all_pts_clone[pcd_idx] -= trans_orig

            # add value in translation direction

            verts[start:end, 0] += (trans_axis[0] * tran_len)
            verts[start:end, 1] += (trans_axis[1] * tran_len)
            verts[start:end, 2] += (trans_axis[2] * tran_len)
            
            # Deal with Point Cloud
            all_pts_clone[pcd_idx] += (trans_axis * tran_len)

            # local coordinate to world coordinate
            verts[start:end, 0] += trans_orig[0]
            verts[start:end, 1] += trans_orig[1]
            verts[start:end, 2] += trans_orig[2]
            
            # Deal with Point Cloud
            all_pts_clone[pcd_idx] += trans_orig
            end_cor = all_pts_clone[pcd_idx,:].clone() 
            
            deformed_data[key+'_'+str(k)] = {
                'verts': verts,
                'faces': faces,
                'all_pts_clone': all_pts_clone,
                'pcd_idx': pcd_idx,
                'start_cor': start_cor,
                'end_cor': end_cor
            }
    
    # no motion
    else:
        return deformed_data
    
    return deformed_data
    

for cad_category in cad_classes:

    folder_path = os.path.join(cad_folder, cad_category)
    if not os.path.exists(folder_path):
        continue
    
    object_paths = [f.path for f in os.scandir(folder_path)]

    for obj_path in object_paths:
        print('processing %s' % obj_path)

        if not os.path.exists(os.path.join(obj_path, 'parts_ply', 'all_pts.txt')):
            continue
        
        # load all points and labels
        with open(os.path.join(obj_path, 'parts_ply', 'all_pts.txt'), 'r') as f:
            all_pts = f.readlines()

        with open(os.path.join(obj_path, 'parts_ply', 'all_label.txt'), 'r') as f:
            all_labels = f.readlines()
            
        all_pts = [x.strip() for x in all_pts]
        all_pts = [x.split(' ') for x in all_pts]
        all_pts = np.array(all_pts).astype(np.float32)
        all_pts = torch.from_numpy(all_pts).float().to(device)
        
        all_labels = [x.strip() for x in all_labels]
        all_labels = [x.split(' ') for x in all_labels]
        all_labels = np.array(all_labels).astype(np.int32)
        all_labels = torch.from_numpy(all_labels).to(torch.int32).to(device)
        
        # load merged mesh and number of vtx for each part
        verts_list, vert_normals_list, vert_textures_list, \
            faces_list, faces_vt_list, faces_vn_list, num_vtx, mtl_files, material_dict = merge_meshes(obj_path)
        
        # make motion directory
        if os.path.exists(os.path.join(obj_path.replace('cad_sapien', 'PartNet-Mobility'), 'motion')):
            shutil.rmtree(os.path.join(obj_path.replace('cad_sapien', 'PartNet-Mobility'), 'motion'))
        os.makedirs(os.path.join(obj_path.replace('cad_sapien', 'PartNet-Mobility'), 'motion'), exist_ok=True)

        
        # copy mtl files
        for mtl_file in mtl_files:
            mtl_path = os.path.join(obj_path, 'parts_ply', mtl_file)
            dst_path = os.path.join(obj_path, 'motion', mtl_file)
            shutil.copy(mtl_path, dst_path)

        # load motion json file
        with open(os.path.join(obj_path, 'motion.json')) as json_file:
            motion = json.load(json_file)

        # at least render one frame
        if len(motion) == 0:
            motion['placeholder'] = {}
        
        idx_key_pairs = [(idx, key) for idx, key in enumerate(motion.keys())]
        # rotate or translate individual part
        for idx, part_key in enumerate(motion.keys()): 
            
            deformed_data = deform_part(idx, part_key, verts_list, all_pts, deform_num=6)
            
            if len(deformed_data) == 0:
                continue
            
            # random select 2 other random parts
            filtered_idx_key_pairs = [(idx, key) for idx, key in idx_key_pairs if key != part_key]
            if len(filtered_idx_key_pairs) >= 2:
                random_pairs = random.sample(filtered_idx_key_pairs, 2)
            else:
                random_pairs = filtered_idx_key_pairs
                    
            for key_1, value_1 in deformed_data.items():
                
                if len(random_pairs) >= 1:
                    # deform other 2 parts, 2 states for each part, 4 states in all (00, 05, 50, 55)
                    deformed_data_part2 = deform_part(random_pairs[0][0], random_pairs[0][1], value_1['verts'], value_1['all_pts_clone'], deform_num=2)
                    
                    for key_2, value_2 in deformed_data_part2.items():
                        
                        if len(random_pairs) >= 2:
                            
                            deformed_data_part3 = deform_part(random_pairs[1][0], random_pairs[1][1], value_2['verts'], value_2['all_pts_clone'], deform_num=2)
                            
                            for key_3, value_3 in deformed_data_part3.items():
                                # Deal with Point Cloud
                                all_pts_clone = torch.stack([value_3['all_pts_clone'][:,2], value_3['all_pts_clone'][:,0], value_3['all_pts_clone'][:,1]], dim=1)
                                verts = torch.stack([value_3['verts'][:,2], value_3['verts'][:,0], value_3['verts'][:,1]], dim=1)
                                start_cor = torch.stack([value_3['start_cor'][:, 2], value_3['start_cor'][:, 0], value_3['start_cor'][:, 1]], dim=1)
                                end_cor = torch.stack([value_3['end_cor'][:, 2], value_3['end_cor'][:, 0], value_3['end_cor'][:, 1]], dim=1) 
                                
                                # save ply and obj
                                save_combined_obj_with_texture(verts, vert_normals_list, vert_textures_list,
                                                faces_list, faces_vt_list, faces_vn_list,
                                                mtl_files, material_dict,
                                                os.path.join(obj_path, 'motion',  key_1 + '_' + key_2.replace('_', '') + key_3.replace('_', '') +'.obj'))
                                
                                # save drag start and end
                                np.save(os.path.join(obj_path, 'motion', key_1.split('_')[0]+'_start.npy'), start_cor.cpu().numpy())
                                np.save(os.path.join(obj_path, 'motion', key_1.split('_')[0]+'_end.npy'), end_cor.cpu().numpy())
                                    
                                np.save(os.path.join(obj_path, 'motion', key_1.split('_')[0] + '_pcd_ind.npy'), value_3['pcd_idx'].cpu().numpy())
                                
                                with open(os.path.join(obj_path, 'motion', key_1.split('_')[0] +'_deform.txt'), 'w') as f:
                                    f.write('Rotation\n')
                                
                                # Save ply
                                xyz = all_pts_clone.cpu().numpy()
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(xyz)
                                o3d.io.write_point_cloud(os.path.join(obj_path, 'motion',  key_1 + '_' + key_2.replace('_', '') + key_3.replace('_', '')  + '.ply'), pcd)
                        else:
                            # Deal with Point Cloud
                            all_pts_clone = torch.stack([value_2['all_pts_clone'][:,2], value_2['all_pts_clone'][:,0], value_2['all_pts_clone'][:,1]], dim=1)
                            verts = torch.stack([value_2['verts'][:,2], value_2['verts'][:,0], value_2['verts'][:,1]], dim=1)
                            start_cor = torch.stack([value_2['start_cor'][:, 2], value_2['start_cor'][:, 0], value_2['start_cor'][:, 1]], dim=1)
                            end_cor = torch.stack([value_2['end_cor'][:, 2], value_2['end_cor'][:, 0], value_2['end_cor'][:, 1]], dim=1) 
                            
                            # save ply and obj
                            save_combined_obj_with_texture(verts, vert_normals_list, vert_textures_list,
                                            faces_list, faces_vt_list, faces_vn_list,
                                            mtl_files, material_dict,
                                            os.path.join(obj_path, 'motion',  key_1 + '_' + key_2.replace('_', '') +'.obj'))
                            
                            # save drag start and end
                            np.save(os.path.join(obj_path, 'motion', key_1.split('_')[0]+'_start.npy'), start_cor.cpu().numpy())
                            np.save(os.path.join(obj_path, 'motion', key_1.split('_')[0]+'_end.npy'), end_cor.cpu().numpy())
                                
                            np.save(os.path.join(obj_path, 'motion', key_1.split('_')[0] + '_pcd_ind.npy'), value_2['pcd_idx'].cpu().numpy())
                            
                            with open(os.path.join(obj_path, 'motion', key_1.split('_')[0] +'_deform.txt'), 'w') as f:
                                f.write('Rotation\n')
                            
                            # Save ply
                            xyz = all_pts_clone.cpu().numpy()
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(xyz)
                            o3d.io.write_point_cloud(os.path.join(obj_path, 'motion', key_1 + '_' + key_2.replace('_', '') + '.ply'), pcd)
                else:    
                    # Deal with Point Cloud
                    all_pts_clone = torch.stack([value_1['all_pts_clone'][:,2], value_1['all_pts_clone'][:,0], value_1['all_pts_clone'][:,1]], dim=1)
                    verts = torch.stack([value_1['verts'][:,2], value_1['verts'][:,0], value_1['verts'][:,1]], dim=1)
                    start_cor = torch.stack([value_1['start_cor'][:, 2], value_1['start_cor'][:, 0], value_1['start_cor'][:, 1]], dim=1)
                    end_cor = torch.stack([value_1['end_cor'][:, 2], value_1['end_cor'][:, 0], value_1['end_cor'][:, 1]], dim=1) 
                    
                    # save ply and obj
                    save_combined_obj_with_texture(verts, vert_normals_list, vert_textures_list,
                                    faces_list, faces_vt_list, faces_vn_list,
                                    mtl_files, material_dict,
                                    os.path.join(obj_path, 'motion',  key_1 + '.obj'))
                    
                    # save drag start and end
                    np.save(os.path.join(obj_path, 'motion', key_1.split('_')[0]+'_start.npy'), start_cor.cpu().numpy())
                    np.save(os.path.join(obj_path, 'motion', key_1.split('_')[0]+'_end.npy'), end_cor.cpu().numpy())
                        
                    np.save(os.path.join(obj_path, 'motion', key_1.split('_')[0] + '_pcd_ind.npy'), value_1['pcd_idx'].cpu().numpy())
                    
                    with open(os.path.join(obj_path, 'motion', key_1.split('_')[0] +'_deform.txt'), 'w') as f:
                        f.write('Rotation\n')
                    
                    # Save ply
                    xyz = all_pts_clone.cpu().numpy()
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz)
                    o3d.io.write_point_cloud(os.path.join(obj_path, 'motion',  key_1 + '.ply'), pcd)
    
    print(f'------------{cad_category} Done---------------')
