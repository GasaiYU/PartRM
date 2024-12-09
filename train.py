import tyro
import time
import random

import torch
from core.models import LGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui

from PIL import Image, ImageDraw
import numpy as np

from torch.utils.tensorboard import SummaryWriter

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
        model.load_state_dict(ckpt, strict=False)
    
    # data
    if dataset_name == 'partdrag4d':
        from core.train_dataset_partdrag4d import PartDrag4DDatset as TrainDataset
        from core.eval_dataset_partdrag4d import PartDrag4DDatset as EvalDataset
    elif dataset_name == 'objavser_hq':
        from core.train_dataset_objaverse_hq import ObjaverseHQDataset as TrainDataset
        from core.eval_dataset_objaverse_hq import ObjaverseHQDataset as EvalDataset

    train_dataset = TrainDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = EvalDataset(opt)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )
    
    writer = SummaryWriter(opt.workspace)

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out, drag_start_2d, drag_move_2d = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                
                writer.add_scalar('Loss/train_iter', loss.item(), epoch * len(train_dataloader) + i)

                accelerator.backward(loss)
                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

                torch.cuda.empty_cache()

            if accelerator.is_main_process:
                # logging
                if i % 100 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                
                # save log images
                if i % 200 == 0:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)


        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
        
        writer.add_scalar('Loss/train_epoch', total_loss.item(), epoch)
        writer.add_scalar('PSNR/train_epoch', total_psnr.item(), epoch)

        accelerator.wait_for_everyone()
        accelerator.save_model(model, opt.workspace)

        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            total_lpips = 0
            for i, data in enumerate(test_dataloader):

                out, drag_start_2d, drag_move_2d = model(data)
    
                psnr = out['psnr']
                lpips = out['loss_lpips']
                total_psnr += psnr.detach()
                total_lpips += lpips.detach()
                
                # save some images
                if accelerator.is_main_process and i % 20 == 0:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)
                    

            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            total_lpips = accelerator.gather_for_metrics(total_lpips).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                total_lpips /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr:.4f} lpips: {total_lpips:.4f}")

            writer.add_scalar('PSNR/eval_epoch', total_psnr.item(), epoch)


if __name__ == "__main__":
    dataset = 'partdrag4d'
    main(dataset)
