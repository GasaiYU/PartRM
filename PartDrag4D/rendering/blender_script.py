"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import mathutils
from typing import IO, Union, Dict, List, Set, Optional, Any

import bpy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str)
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
    
parser.add_argument("--scale", type=float, default=1)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist", type=int, default=2.4)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"


def adjust_focus_to_head(object):
    bbox_min, bbox_max = scene_bbox(object)
    focus_point = (bbox_min + bbox_max) / 2
    focus_point.z =  bbox_min.z 

    return focus_point

def scene_bbox(single_obj=None):
    """Calculate the bounding box of a single object or the entire scene."""
    bbox_min = Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))
    if single_obj:
        objects = [single_obj]
    else:
        objects = bpy.context.scene.objects

    for obj in objects:
        if obj.type == 'MESH':
            for v in obj.bound_box:
                v_world = obj.matrix_world @ Vector(v)
                bbox_min = Vector(map(min, zip(bbox_min, v_world)))
                bbox_max = Vector(map(max, zip(bbox_max, v_world)))

    return bbox_min, bbox_max

def view_plane(camd, winx, winy, xasp, yasp):    
    #/* fields rendering */
    ycor = yasp / xasp
    use_fields = False
    if (use_fields):
      ycor *= 2

    def BKE_camera_sensor_size(p_sensor_fit, sensor_x, sensor_y):
        #/* sensor size used to fit to. for auto, sensor_x is both x and y. */
        if (p_sensor_fit == 'VERTICAL'):
            return sensor_y

        return sensor_x

    if (camd.type == 'ORTHO'):
      #/* orthographic camera */
      #/* scale == 1.0 means exact 1 to 1 mapping */
      pixsize = camd.ortho_scale
    else:
      #/* perspective camera */
      sensor_size = BKE_camera_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
      pixsize = (sensor_size * camd.clip_start) / camd.lens

    #/* determine sensor fit */
    def BKE_camera_sensor_fit(p_sensor_fit, sizex, sizey):
        if (p_sensor_fit == 'AUTO'):
            if (sizex >= sizey):
                return 'HORIZONTAL'
            else:
                return 'VERTICAL'

        return p_sensor_fit

    sensor_fit = BKE_camera_sensor_fit(camd.sensor_fit, xasp * winx, yasp * winy)

    if (sensor_fit == 'HORIZONTAL'):
      viewfac = winx
    else:
      viewfac = ycor * winy

    pixsize /= viewfac

    #/* extra zoom factor */
    pixsize *= 1 #params->zoom

    #/* compute view plane:
    # * fully centered, zbuffer fills in jittered between -.5 and +.5 */
    xmin = -0.5 * winx
    ymin = -0.5 * ycor * winy
    xmax =  0.5 * winx
    ymax =  0.5 * ycor * winy

    #/* lens shift and offset */
    dx = camd.shift_x * viewfac # + winx * params->offsetx
    dy = camd.shift_y * viewfac # + winy * params->offsety

    xmin += dx
    ymin += dy
    xmax += dx
    ymax += dy

    #/* the window matrix is used for clipping, and not changed during OSA steps */
    #/* using an offset of +0.5 here would give clip errors on edges */
    xmin *= pixsize
    xmax *= pixsize
    ymin *= pixsize
    ymax *= pixsize

    return xmin, xmax, ymin, ymax


def projection_matrix(camd):
    r = bpy.context.scene.render
    left, right, bottom, top = view_plane(camd, r.resolution_x, r.resolution_y, 1, 1)

    farClip, nearClip = camd.clip_end, camd.clip_start

    Xdelta = right - left
    Ydelta = top - bottom
    Zdelta = farClip - nearClip

    mat = [[0]*4 for i in range(4)]

    mat[0][0] = nearClip * 2 / Xdelta
    mat[1][1] = nearClip * 2 / Ydelta
    mat[2][0] = (right + left) / Xdelta #/* note: negate Z  */
    mat[2][1] = (top + bottom) / Ydelta
    mat[2][2] = -(farClip + nearClip) / Zdelta
    mat[2][3] = -1
    mat[3][2] = (-2 * nearClip * farClip) / Zdelta

    return sum([c for c in mat], [])

def set_camera_focus_on_head(camera, model_object):
    focus_point = Vector((-0.0016885548830032349, -0.007119309157133102, -0.38998815417289734))    
    empty = bpy.data.objects.new("Empty", None)
    empty.location = focus_point
    bpy.context.scene.collection.objects.link(empty)
    camera.constraints.clear()
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    projection_matrix_value = projection_matrix(camera.data)
    print(projection_matrix_value)
    

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def load_object(filepath: str):
    if filepath.endswith(".glb") or filepath.endswith(".vrm"):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif filepath.endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=filepath)
    elif filepath.endswith(".obj"):
        if filepath.endswith(".obj"):
            obj_path = filepath
            dirpath = os.path.dirname(filepath)
        else:
            obj_path = os.path.join(filepath, "model.obj")
            dirpath = filepath
        base_name = os.path.basename(obj_path)
        bpy.ops.import_scene.obj(
            filepath=obj_path,
            axis_forward='Y',
            axis_up='Z',
        )
    elif filepath.endswith(".vrm"):
        bpy.ops.import_scene.vrm(filepath=filepath)
 

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def get_3x4_RT_matrix_from_blender(cam):
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def customized_normalize_scene():
    scale =  0.42995866893870127
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()
    # Apply scale to matrix_world.
    offset = Vector((-0.0121, -0.0070, 0.0120))
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")
    

def generate_spherical_spiral_camera_views(num_cameras, radius, num_spirals):
    spirals = np.linspace(0, num_spirals * 2 * np.pi, num_cameras)
    height = np.ones((num_cameras,)) * 1.5
    radius_at_spiral = np.sqrt(radius**2 - height**2)
    x = radius_at_spiral * np.cos(spirals)
    y = radius_at_spiral * np.sin(spirals)
    z = height
    return x, y, z


def set_spherical_spiral_camera(camera, index, num_cameras, radius, num_spirals):
    x, y, z = generate_spherical_spiral_camera_views(num_cameras, radius, num_spirals)
    camera.location = (x[index], y[index], z[index])
    camera.rotation_mode = 'XYZ'
    look_direction = Vector((0, 0, 0)) - Vector((x[index], y[index], z[index]))
    camera.rotation_euler = look_direction.to_track_quat('-Z', 'Y').to_euler()
    c2w_matrix = camera.matrix_world
    bpy.context.view_layer.update()

    camera_matrix = {
        "origin": c2w_matrix.translation[:], 
        "x": c2w_matrix.col[0][:3],
        "y": c2w_matrix.col[1][:3],
        "z": c2w_matrix.col[2][:3],
    }
    return camera_matrix
    
def get_local2world_mat(blender_obj) -> np.ndarray:
    """Returns the pose of the object in the form of a local2world matrix.
    :return: The 4x4 local2world matrix.
    """
    obj = blender_obj
    # Start with local2parent matrix (if obj has no parent, that equals local2world)
    matrix_world = obj.matrix_basis

    # Go up the scene graph along all parents
    while obj.parent is not None:
        # Add transformation to parent frame
        matrix_world = (
            obj.parent.matrix_basis @ obj.matrix_parent_inverse @ matrix_world
        )
        obj = obj.parent

    return np.array(matrix_world)
    
def save_images(object_file: str, args):
    os.makedirs(args.output_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.resolution_x = 512   
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.resolution_percentage = 100 

    bpy.context.scene.render.film_transparent = True

    
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'


    bpy.context.scene.world = bpy.data.worlds.new("WhiteWorld")
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (1, 1, 1, 0)  # RGBA
    
    # Load the object and normalize the scene (implement these functions based on your needs)
    load_object(object_file)  # Assuming this imports your VRM/3D model
    model_object = bpy.context.view_layer.objects.active 

    customized_normalize_scene()
    object_uid = object_file.split("/")[-3] + '_' + os.path.basename(object_file).split(".")[0]
    
    cam = bpy.data.cameras.new('Camera')
    cam_ob = bpy.data.objects.new('Camera', cam)
    bpy.context.scene.collection.objects.link(cam_ob)
    bpy.context.scene.camera = cam_ob
    
    set_camera_focus_on_head(cam_ob, model_object)  # Set the camera focus


    # Setup basic lighting
    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (5, 5, 5)

    num_cameras = args.num_images
    radius = args.camera_dist
    num_spirals = 1

    for i in range(num_cameras):
        camera_matrix = set_spherical_spiral_camera(bpy.context.scene.camera, i, num_cameras, radius, num_spirals)
        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        bpy.context.scene.render.filepath = render_path
        bpy.context.view_layer.use_pass_z = False
        bpy.context.scene.use_nodes = False
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        
        bpy.ops.render.render(write_still=True)
        
        camera_params_path = os.path.join(args.output_dir, object_uid, f"{i:03d}_camera.json")
        camera_params = {
            "origin": list(camera_matrix["origin"]),
            "x": list(camera_matrix["x"]),
            "y": list(camera_matrix["y"]),
            "z": list(camera_matrix["z"]),
        }
        
        # output matrix 
        view_matrix = bpy.context.scene.camera.matrix_world.inverted()
        projection_matrix_b = bpy.context.scene.camera.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(), 
            x=512,
            y=512,
            scale_x=bpy.context.scene.render.pixel_aspect_x,
            scale_y=bpy.context.scene.render.pixel_aspect_y,
        )
        
        with open(camera_params_path, 'w') as f:
            json.dump(camera_params, f, indent=4)


if __name__ == "__main__":
    try:
        start_i = time.time()
        local_path = args.object_path
        save_images(local_path, args)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
