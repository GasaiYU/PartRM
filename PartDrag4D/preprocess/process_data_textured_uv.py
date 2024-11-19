# Copyright (c) Facebook, Inc. and its affiliates.
import math
import os
import numpy as np
# from skimage import img_as_ubyte
from tqdm import tqdm
import re
import open3d as o3d
import itertools
import json
import csv
import shutil

PARTNET_FOLDER = '../data/partnet-mobility-v0-dataset'
OUT_FOLDER = '../data/processed_data_partdrag4d'


# helper function for computing roation matrix in 3D
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# helper function for traversing a tree
def traverse_tree(current_node, mesh_dict):
    # further traverse the tree if not at leaf node yet
    if 'children' in current_node.keys():
        for idx in range(len(current_node['children'])):
            traverse_tree(current_node['children'][idx], mesh_dict)
    else:
        # insert meshes associated with an unique part id
        if 'objs' in current_node.keys():
            assert current_node['id'] not in mesh_dict.keys()
            mesh_dict[current_node['id']] = current_node['objs']
        else:
            print(f"Empty node {part_folder}, id {current_node['id']}. Skipping.")
    return

# helper function for merging mtl files into one and renaming materials
def merge_and_rename_materials(src_mtl_path, dst_mtl_path, material_ranges, existing_material_count):

    material_count = 0
    updated_material_ranges = {}
    
    with open(src_mtl_path, 'r') as src_file:
        lines = src_file.readlines()
    
    with open(dst_mtl_path, 'a') as dst_file:
        write_block = False
        
        for line in lines:
            if line.startswith('newmtl'):
                cur_material_name = line.strip().split()[1]
                if cur_material_name in material_ranges:
                    obj_count = os.path.splitext(os.path.basename(dst_mtl_path))[0]
                    new_material_name = 'material-' + str(obj_count) + '-' + str(material_count + existing_material_count)
                    material_count += 1
                    
                    # rename material range dict
                    updated_material_ranges[new_material_name] = material_ranges[cur_material_name]
                    
                    # activate writing to merged mtl file
                    dst_file.write(f'newmtl {new_material_name}\n')
                    write_block = True
                else:
                    write_block = False
                    
            elif write_block:
                dst_file.write(line)
    
    return updated_material_ranges

# helper function for loading obj with vn, vt
def load_obj_with_texture(obj_path):
    # store verts and faces
    verts = []
    vertex_normals = []
    vertex_textures = []
    faces = {}
    faces.setdefault('v', [])
    faces.setdefault('vt', [])
    faces.setdefault('vn', [])
    
    # store material
    mtl_files = []
    material_ranges = {}
    prev_material_name = None 
    face_count = 0 
    material_name = None
    
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            # load verts and faces
            if line.startswith('v '):
                v = [float(x) for x in line.strip().split()[1:]]
                verts.append(v)
            elif line.startswith('vn '):
                vn = [float(x) for x in line.strip().split()[1:]]
                vertex_normals.append(vn)
            elif line.startswith('vt '):
                vt = [float(x) for x in line.strip().split()[1:]]
                vertex_textures.append(vt)
            elif line.startswith('f '):
                f = line.strip().split()[1:]
                f_v = []
                f_vt = []
                f_vn = []
                for face_vertex in f:
                    split_id = face_vertex.split('/')
                    if len(split_id) == 3:
                        f_v.append(int(split_id[0]))
                        f_vt.append(int(split_id[1]) if split_id[1] != '' else None)
                        f_vn.append(int(split_id[2]))
                    else:
                        f_v.append(int(split_id[0]))
                        f_vt.append(None)
                        f_vn.append(None)
                faces['v'].append(f_v)
                faces['vt'].append(f_vt)
                faces['vn'].append(f_vn)
                face_count += 1
            
            # load material
            elif line.startswith('mtllib'):
                mtl_files.append(line.strip().split()[1])
            
            # store current material starting face and previous material ending face
            elif line.startswith('usemtl'):
                current_material = {}
                material_name = line.strip().split()[1]
                current_material['start'] = face_count
                material_ranges[material_name] = current_material
                if prev_material_name is not None:
                    material_ranges[prev_material_name]['end'] = face_count -1
                prev_material_name = material_name
        
        # last material ending face
        if material_name is not None:
            material_ranges[material_name]['end'] = face_count -1
            
    return verts, vertex_normals, vertex_textures, faces, mtl_files, material_ranges
    
    
# helper function for loading obj with mtl file name and face ranges of used materials 
def load_obj_with_material(obj_path):
    # load materials
    mtl_files = []
    material_ranges = {}
    prev_material_name = None 
    face_count = 0 
    
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('mtllib'):
                mtl_files.append(line.strip().split()[1])
            
            # store current material starting face and previous material ending face
            elif line.startswith('usemtl'):
                current_material = {}
                material_name = line.strip().split()[1]
                current_material['start'] = face_count
                material_ranges[material_name] = current_material
                if prev_material_name is not None:
                    material_ranges[prev_material_name]['end'] = face_count -1
                prev_material_name = material_name
            
            # count faces
            elif line.startswith('f'):
                face_count += 1
        
        # last material ending face
        material_ranges[material_name]['end'] = face_count -1

    # load verts and faces
    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    return mesh, mtl_files, material_ranges

# helper function for writing verts and faces with asigned material name
def save_combined_obj(verts, normals, faces, mtl_files, mtl_dict, output_obj):
    # skip empty obj
    if len(verts) == 0:
        return
    
    with open(output_obj, 'w') as file:
        for mtl_file in mtl_files:
            file.write(f'mtllib {os.path.basename(mtl_file)}\n')
        for v in verts:
            file.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for vn in normals:
            file.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))
        for mtl_name, mtl_range in mtl_dict.items():
            file.write(f'usemtl {mtl_name}\n')
            for i in range(mtl_range['start'], mtl_range['end'] + 1):
                file.write('f %d//%d %d//%d %d//%d\n' % (faces[i, 0], faces[i, 0], faces[i, 1], faces[i, 1], faces[i, 2], faces[i, 2]))

# helper function for writing verts and faces with asigned material name
def save_combined_obj_with_texture(verts, normals, textures, faces, faces_vt, faces_vn, mtl_files, mtl_dict, output_obj):
    # skip empty obj
    if len(verts) == 0:
        return
    
    with open(output_obj, 'w') as file:
        for mtl_file in mtl_files:
            file.write(f'mtllib {os.path.basename(mtl_file)}\n')
        for v in verts:
            file.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for vn in normals:
            file.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))
        for vt in textures:
            file.write('vt %f %f\n' % (vt[0], vt[1]))
        for mtl_name, mtl_range in mtl_dict.items():
            file.write(f'usemtl {mtl_name}\n')
            for i in range(mtl_range['start'], mtl_range['end'] + 1):
                # if faces are loaded by o3d, need to +1 to each number
                file.write('f {}/{}/{} {}/{}/{} {}/{}/{}\n'.format(
                    str(int(faces[i, 0])), str(faces_vt[i][0]) if faces_vt[i][0] is not None else '', str(faces_vn[i][0]) if faces_vn[i][0] is not None else '',
                    str(int(faces[i, 1])), str(faces_vt[i][1]) if faces_vt[i][1] is not None else '', str(faces_vn[i][1]) if faces_vn[i][1] is not None else '',
                    str(int(faces[i, 2])), str(faces_vt[i][2]) if faces_vt[i][2] is not None else '', str(faces_vn[i][2]) if faces_vn[i][2] is not None else ''))

# helper function for loading and merging meshes
def merge_meshes(save_folder, ids, mesh_dict):
    for count, part_ids in enumerate(ids):
        part_meshes = [mesh_dict[x] for x in part_ids]
        part_meshes = list(itertools.chain(*part_meshes))
        
        verts_list = np.empty((0,3))
        vert_normals_list = np.empty((0,3))
        vert_textures_list = np.empty((0,2))
        faces_list = np.empty((0,3))#.long()
        faces_vt_list = []
        faces_vn_list = []
        material_dict = {}
        
        save_path = os.path.join(save_folder, 'parts_ply')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        merged_mtl_path = os.path.join(save_path, str(count)+'.mtl')
        if os.path.exists(merged_mtl_path):
            os.remove(merged_mtl_path)

        for part_mesh in part_meshes:
            obj_path = os.path.join(part_folder, 'textured_objs', part_mesh,)+'.obj'
            # check if mesh exist
            if not os.path.exists(obj_path):
                print(f"Missing {obj_path}")
                continue
                        
            # load texture
            verts, vertex_normals, vertex_textures, faces, mtl_files, material_ranges = load_obj_with_texture(obj_path)
            
            verts = np.asarray(verts)
            vertex_normals = np.asarray(vertex_normals)
            vertex_textures = np.asarray(vertex_textures)
            
            # reindex faces and update material face indices
            faces_v = [[val + verts_list.shape[0] for val in sublist] for sublist in faces['v']]
            for _, face_range in material_ranges.items():
                face_range['start'] += faces_list.shape[0]
                face_range['end'] += faces_list.shape[0]
            
            # reindex face coordinates
            faces_vt = [[val if val is None else val + vert_textures_list.shape[0]
                         for val in sublist] for sublist in faces['vt']]
            faces_vn = [[val if val is None else val + vert_normals_list.shape[0]
                         for val in sublist] for sublist in faces['vn']]
            
            # merge mtl files and rename materials
            assert len(mtl_files) == 1
            src_mtl_path = os.path.join(part_folder, 'textured_objs', mtl_files[0])
            materials = merge_and_rename_materials(src_mtl_path, merged_mtl_path, material_ranges, len(material_dict))
            material_dict.update(materials)
            
            # merge verts and faces
            verts_list = np.concatenate([verts_list, verts])
            vert_normals_list = np.concatenate([vert_normals_list, vertex_normals])
            if vertex_textures.shape[0] != 0:
                vert_textures_list = np.concatenate([vert_textures_list, vertex_textures])
            faces_list = np.concatenate([faces_list, faces_v])
            faces_vt_list.extend(faces_vt)
            faces_vn_list.extend(faces_vn)
        
        # merge obj file with renamed materials
        dst_obj_path =  os.path.join(save_path, str(count)+'.obj')
        merged_mtl_file = [] 
        merged_mtl_file.append(merged_mtl_path)
        save_combined_obj_with_texture(verts_list, vert_normals_list, vert_textures_list,
                                       faces_list, faces_vt_list, faces_vn_list,
                                       merged_mtl_file, material_dict, dst_obj_path)


if __name__ == "__main__":

    part_home = PARTNET_FOLDER 
    save_home = OUT_FOLDER 
    classes = ['Dishwasher', 'Laptop', 'Microwave', 'Oven', 
               'Refrigerator', 'StorageFurniture', 'WashingMachine', 'TrashCan']

    count = 0

    # all dirIDs within this class
    with open('../filelist/partdrag4d.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['category'] in classes:
                part_dir = row['category']
                part_id = row['id']
                # part_id = 9992
                part_folder = os.path.join(part_home, str(part_id))
                save_folder = os.path.join(save_home, part_dir, str(part_id))
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                count+=1

                # copy textured images
                if 'images' in os.listdir(part_folder):
                    src_images_folder = os.path.join(part_folder, 'images')
                    dest_images_folder = os.path.join(save_folder, 'images')
                    if os.path.exists(dest_images_folder):
                        shutil.rmtree(dest_images_folder)
                    shutil.copytree(src_images_folder, dest_images_folder)
                    
                # load meshes referenced json file
                if not os.path.isfile(os.path.join(part_folder, 'result.json')):
                    continue  
                with open(os.path.join(part_folder, 'result.json')) as json_file:
                    part_meshes = json.load(json_file)
                    # traverse through a tree
                    mesh_dict = {}
                    root = part_meshes[0]
                    traverse_tree(root, mesh_dict)

                types = []
                with open(os.path.join(part_folder, 'mobility.urdf')) as f:
                    our_lines = f.readlines()
                    for line in our_lines:
                        myString = re.sub('\s+',' ',line)
                        if '<joint name=' in myString:
                            m_type = myString.split("type=",1)[1][1:-3]
                            types.append(m_type)
                type_idx = 0
                details = {}
                details_saved = {}

                # load mobility_v2 json file
                with open(os.path.join(part_folder, 'mobility_v2.json')) as json_file:

                    mobility_parts = json.load(json_file)
                    print('processing %s' % part_folder)

                    part_div = []
                    for idx, joint_part in enumerate(mobility_parts):

                        # visual names belonging to one joint part
                        joint_part_names = joint_part['parts']
                        assert(joint_part_names) # make sure not empty

                        # parse ids for each part
                        ids = [x['id'] for x in joint_part_names]

                        part_div.append(ids)

                        # save motion information
                        details[str(idx)] = joint_part['jointData'].copy()
                        details_saved[str(idx)] = joint_part['jointData'].copy()
                        
                        # set type for care part
                        if type_idx<len(types):
                            if True:
                                details[str(idx)]['type'] = types[type_idx]
                                details_saved[str(idx)]['type'] = types[type_idx]
                                details_saved[str(idx)]['part_div'] = part_div[-1]
                            type_idx += 1
                        else:
                            if details[str(idx)]:
                                assert type_idx>=len(types)
                                # assert joint_part['name'] not in careParts[part_dir]

                        # remove non-care part
                        if not joint_part['jointData']:
                            details[str(idx)] = {}
                            details_saved.pop(str(idx), None)

                    with open(os.path.join(save_folder, 'motion.json'), 'w') as outfile:
                        json.dump(details_saved, outfile)

                    assert len(details) == len(part_div)
                    part_idx = 0
                    fix_part = []
                    parts = []

                    for key, value in details.items():
                        if value == {}:
                            fix_part.append(part_div[part_idx])
                        else:
                            parts.append(part_div[part_idx])
                        part_idx += 1
                        
                    fix_part = list(itertools.chain(*fix_part))
                    parts.append(fix_part)

                    # load, merge, and save part mesh file
                    merge_meshes(save_folder, parts, mesh_dict)

                    if os.path.exists(os.path.join(part_folder, 'point_sample', 'label-10000.txt')) and os.path.exists(os.path.join(part_folder, 'point_sample', 'pts-10000.txt')):
                        shutil.copy(os.path.join(part_folder, 'point_sample', 'pts-10000.txt'), os.path.join(save_folder, 'parts_ply', 'all_pts.txt'))
                        shutil.copy(os.path.join(part_folder, 'point_sample', 'label-10000.txt'), os.path.join(save_folder, 'parts_ply', 'all_label.txt'))
    print(count)
    print('all done...')
