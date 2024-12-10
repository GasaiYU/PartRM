import os
import numpy as np
from PIL import Image

import argparse

def gen_rgba(image_path):
    """
    Generate rgba images from rgb images which are with white background.
    Args:
        image_path (str): Path to the rgb image.
    """
    image = Image.open(image_path)
    rgba_image = image.convert('RGBA')
    pixels = rgba_image.load()

    for y in range(rgba_image.height):
        for x in range(rgba_image.width):
            r, g, b, a = pixels[x, y]

            if r >= 245 and g >= 245 and b >= 245:
                pixels[x, y] = (r, g, b, 0)

    rgba_image = rgba_image.resize((512, 512))
    rgba_image.save(image_path.replace('.png', '_rgba.png'))    

def gen_rgba_from_filelist(file_list_path, dataset_name):
    """
    Generate rgba images from rgb images which are with white background.
    Args:
        file_list_path (str): Path to the file list.
    """
    with open(file_list_path, 'r') as f:
        file_list = f.readlines()

    if dataset_name == 'partdrag4d':
        for file_name in file_list:
            file_name = file_name.strip()
            for image_path in os.listdir(file_name):
                if image_path.endswith('.png') and not image_path.endswith('_rgba.png') and not image_path.startswith('000'):
                    image_path = os.path.join(file_name, image_path)
                    gen_rgba(image_path)

    elif dataset_name == 'objaverse_hq':
        for action_dir in file_list:
            action_dir = action_dir.strip()
            for frame_dir in os.listdir(action_dir):
                frame_dir = os.path.join(action_dir, frame_dir)
                for image_path in os.listdir(frame_dir):
                    if image_path.endswith('.png') and not image_path.endswith('_rgba.png') and not image_path.startswith('000'):
                        image_path = os.path.join(frame_dir, image_path)
                        gen_rgba(image_path)
            
    else:
        raise ValueError('Unknown dataset name.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', type=str, help='path/to/your/zero123/filelist')
    parser.add_argument('--dataset', choices=['partdrag4d', 'objaverse_hq'], help='dataset name')
    args = parser.parse_args()
    
    gen_rgba_from_filelist(args.filelist, args.dataset)