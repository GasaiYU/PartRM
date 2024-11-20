import tyro
import torch
from core.models import LGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui
import numpy as np

import argparse


def main(dataset_name):    
    if dataset_name == 'partdrag4d':
        from core.options import AllConfigs
    else:
        from core.options_pm import AllConfigs

    opt = tyro.cli(AllConfigs)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    model = LGM(opt)
    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        model.load_state_dict(ckpt, strict=False)

    # data
    if dataset_name == 'partdrag4d':
        from core.eval_dataset_partdrag4d import PartDrag4DDatset as EvalDataset
    elif dataset_name == 'objavser_hq':
        from core.eval_dataset_objaverse_hq import ObjaverseHQDataset as EvalDataset

    test_dataset = EvalDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # accelerate
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    # loop
    for epoch in range(1):
        # eval
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(test_dataloader):
                out, drag_start_2d, drag_move_2d = model(data)
                
                # save some images
                if accelerator.is_main_process:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                    origin_images = data['images_input'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    origin_images = origin_images.transpose(0, 3, 1, 4, 2).reshape(-1, origin_images.shape[1] * origin_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_origin_images_{epoch}_{i}.jpg', origin_images)
                
            torch.cuda.empty_cache()

if __name__ == "__main__":
    dataset = 'partdrag4d'
    main(dataset)