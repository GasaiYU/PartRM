import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from zero123plus.model import MVDiffusion

from torchvision.utils import make_grid, save_image

import torchvision.transforms as TF

import os
import numpy as np

import argparse

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image

def main(src_filelist, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Load the pipeline
    mv_diffusion = MVDiffusion({'pretrained_model_name_or_path':"sudo-ai/zero123plus-v1.2",
                            'custom_pipeline':"./zero123plus"})
    state_dict = torch.load('/gpfs/essfs/iat/Tsinghua/gaomx/gaomx/workspace/blenderRender/lgm_arti/third_party/InstantMesh/logs/zero123plus-finetune-part/checkpoints/step=00005000.ckpt')['state_dict']

    mv_diffusion.load_state_dict(state_dict, strict=True)
    pipeline = mv_diffusion.pipeline
    pipeline = pipeline.to('cuda')

    val_image_paths = []
    with open(src_filelist, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            image_path = f"{line}/000.png"
            val_image_paths.append(image_path)
    print(f"The length of the val images: {len(val_image_paths)}")

    for val_image_path in val_image_paths:
        print(f'Render {val_image_path}')
        render_id = val_image_path.split('/')[-2]

        cond = Image.open(val_image_path)
        os.makedirs(os.path.join(save_dir, render_id))
        cond.save(os.path.join(save_dir, render_id, '000.png'))

        # Run the pipeline!
        with torch.no_grad():
            latents = pipeline(cond, num_inference_steps=75, output_type='latent').images
            images = unscale_image(pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0])   # [-1, 1]
            images = (images * 0.5 + 0.5).clamp(0, 1)

            resize = TF.Resize((512, 512))
            images = images.squeeze(0)
            for i in range(6):
                row_idx = i % 2
                col_idx = i // 2
                image = images[:, col_idx*320:col_idx*320+320, row_idx*320:row_idx*320+320]
                image = resize(image)
                save_image(image, os.path.join(save_dir, render_id, f'{i+1:03d}.png'))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_filelist', default='../filelist/test.txt')
    parser.add_argument('--output_dir', default='./zero123_preprocessed_data/PartDrag4D')
    args = parser.parse_args()

    main(args.src_filelist, args.output_dir)
