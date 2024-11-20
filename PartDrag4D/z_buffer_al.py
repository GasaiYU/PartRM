import open3d as o3d

import numpy as np
import os

from tqdm import tqdm

import argparse

view_matrix = np.array([[-3.7966e-03,  7.0987e-01, -7.0432e-01,  1.8735e+00],
        [ 9.9999e-01,  2.6951e-03, -2.6740e-03,  0.0000e+00],
        [-1.1642e-10, -7.0432e-01, -7.0988e-01,  1.5000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

def local2world(points, scale, world_matrix):
    points = points.copy()
    scale = np.array(scale).copy()
    translation = np.array(world_matrix, dtype=np.float32)
    
    scaled_points = points * scale
    world_points = scaled_points + translation
    
    return world_points

def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def judge(center, radius, z_buffer, cur_depth, epsilon=0.005):
    var_x = [max(center[0] - radius, 0), min(center[0] + radius, 255)]
    var_y = [max(center[1] - radius, 0), min(center[1] + radius, 255)]

    for i in range(var_x[0], var_x[1]):
        for j in range(var_y[0], var_y[1]):
            if z_buffer[i, j] == np.inf:
                continue
            if z_buffer[i, j] > cur_depth + epsilon:
                return False
    return True

def refine_z_buffer_index(z_buffer, z_buffer_index, pcd_points):
    for i in range(z_buffer.shape[0]):
        for j in range(z_buffer.shape[1]):
            if z_buffer_index[i, j] == 0:
                continue
            if not judge([i, j], 10, z_buffer, z_buffer[i, j]):
                z_buffer_index[i, j] = 0
                z_buffer[i, j] = np.inf


def project_points(points, cam_view):
    # local2world transformation using blender scale and world matrix
    scale =  0.42995866893870127
    world_matrix = (-0.0121, -0.0070, 0.0120)
    points = local2world(points, scale, world_matrix)
    
    # use blender fixed projection matrix
    proj_matrix = [
         [2.777777671813965, 0.0000,  0.0000,  0.0000],
            [0.0000, 2.777777671813965,  0.0000,  0.0000],
            [0.0000, 0.0000, -1.0001999139785767, -0.20002000033855438],
            [0.0000, 0.0000, -1.0000,  0.0000]
    ]
    cam_proj = np.array(proj_matrix, dtype=np.float32)
    
    z_buffer = np.full((256, 256), fill_value=float('inf'))
    z_buffer_index = np.zeros([256, 256], dtype=np.int32)
    
    S = np.array([256, 256])

    for i in range(points.shape[0]):
        view_matrix = cam_view
        view_matrix = np.linalg.inv(view_matrix)
        proj_matrix = cam_proj
        
        point_3D = points[i]
        point_3D_homogeneous = np.concatenate([point_3D, np.array([1.0])], axis=0)
        
        # Transfer to camera coordinates
        camera_coords = np.matmul(view_matrix, point_3D_homogeneous)
        
        # Transfer to clip coordinates
        clip_coords = np.matmul(proj_matrix, camera_coords)

        ndc_coords = clip_coords[:3] / clip_coords[3]
        
        ndc_coords_2d = ndc_coords[:2]
        ndc_depth = ndc_coords[2]

        # Get screen coordinates
        screen_coords = S - ndc2Pix(ndc_coords_2d, S)
        
        # Check if in the screen
        if int(screen_coords[0]) < 0 or int(screen_coords[0]) >= 256 or int(screen_coords[1]) < 0 or int(screen_coords[1]) >= 256:
            continue

        # z-buffer
        if z_buffer[int(screen_coords[0]), int(screen_coords[1])] > ndc_depth:
            z_buffer[int(screen_coords[0]), int(screen_coords[1])] = ndc_depth
            z_buffer_index[int(screen_coords[0]), int(screen_coords[1])] = i

    refine_z_buffer_index(z_buffer, z_buffer_index, points)
    return np.unique(z_buffer_index)

def main(mesh_base):
    for categorial in os.listdir(mesh_base):
        if not os.path.isdir(os.path.join(mesh_base, categorial)):
            continue
        for mesh_id in tqdm(os.listdir(os.path.join(mesh_base, categorial))):
            if not os.path.exists(os.path.join(mesh_base, categorial, mesh_id, "motion")):
                continue
            for ply_file in os.listdir(os.path.join(mesh_base, categorial, mesh_id, "motion")):
                if ply_file.endswith(".ply"):
                    print(f"Process {os.path.join(mesh_base, categorial, mesh_id, 'motion', ply_file)}")
                    pcd = o3d.io.read_point_cloud(os.path.join(mesh_base, categorial, mesh_id, "motion", ply_file))
                    z_buffer_index = project_points(np.array(pcd.points), view_matrix)
                    np.save(os.path.join(mesh_base, categorial, mesh_id, "motion", ply_file.replace(".ply", "_visible.npy")), z_buffer_index)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_base', default='./data/processed_data_partdrag4d', help='The render images dir')
    args = parser.parse_args()

    main(args.mesh_base)
                    

    
    
    