import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file

import kiui
from kiui.lpips import LPIPS

from core.unet import UNet, UNetWithMSDrag
from core.options import Options
from core.gs import GaussianRenderer

from core.unet_original import UNet as UNetOriginal



class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        if opt.use_ms_drag_encoding:
            self.unet = UNetWithMSDrag(
                9, 14, 
                down_channels=opt.down_channels,
                down_attention=opt.down_attention,
                mid_attention=opt.mid_attention,
                up_channels=opt.up_channels,
                up_attention=opt.up_attention,
                use_drag_encoding=opt.use_drag_encoding
            )
        else:
            self.unet = UNet(
                9, 14, 
                down_channels=self.opt.down_channels,
                down_attention=self.opt.down_attention,
                mid_attention=self.opt.mid_attention,
                up_channels=self.opt.up_channels,
                up_attention=self.opt.up_attention,
                use_drag_encoding=self.opt.use_drag_encoding
            )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again


        # Original Unet
        if self.opt.add_with_original:
            self.unet_original = UNetOriginal(
                9, 14, 
                down_channels=self.opt.down_channels,
                down_attention=self.opt.down_attention,
                mid_attention=self.opt.mid_attention,
                up_channels=self.opt.up_channels,
                up_attention=self.opt.up_attention,
            )
            self.conv_original = nn.Conv2d(14, 14, kernel_size=1)
            ckpt = load_file(self.opt.original_ckpt, device='cpu')
            unet_state_dict = self.unet_original.state_dict()
            conv_state_dict = self.conv_original.state_dict()

            for k, v in ckpt.items():
                new_k = '.'.join(k.split('.')[1:])
                if k.split('.')[0] == 'unet' and new_k in unet_state_dict:
                    if unet_state_dict[new_k].shape == v.shape:
                        unet_state_dict[new_k].copy_(v)
                    else:
                        print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {unet_state_dict[k].shape}, ignored.')
                elif k.split('.')[0] == 'conv' and new_k in conv_state_dict:
                    if conv_state_dict[new_k].shape == v.shape:
                        conv_state_dict[new_k].copy_(v)
                    else:
                        print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {conv_state_dict[k].shape}, ignored.')
                else:
                    print(f'[WARN] unexpected param {k}: {v.shape}')

            if self.opt.stage1:
                for param in self.unet_original.parameters():
                    param.requires_grad = False
                for param in self.conv_original.parameters():
                    param.requires_grad = False
                self.unet_original.eval()
                self.conv_original.eval()

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
        

    def forward_gaussians(self, images, drags_start=None, drags_end=None):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]
        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images, drags_start, drags_end) # [B*4, 14, h, w]
        x = self.conv(x) # [B*4, 14, h, w]

        x = x.reshape(B, 4, 14, self.opt.splat_size, self.opt.splat_size)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)

        if self.opt.add_with_original:
            if self.opt.stage1:
                with torch.no_grad():
                    y = self.unet_original(images)
                    y = self.conv_original(y)
                    y = y.reshape(B, 4, 14, self.opt.splat_size, self.opt.splat_size)
                    y = y.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
            else:
                y = self.unet_original(images)
                y = self.conv_original(y)
                y = y.reshape(B, 4, 14, self.opt.splat_size, self.opt.splat_size)
                y = y.permute(0, 1, 3, 4, 2).reshape(B, -1, 14) 

        if not self.opt.add_with_original:
            pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
            opacity = self.opacity_act(x[..., 3:4])
            scale = self.scale_act(x[..., 4:7])
            rotation = self.rot_act(x[..., 7:11])
            rgbs = self.rgb_act(x[..., 11:])
        else:
            x = x + y
            pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
            opacity = self.opacity_act(x[..., 3:4])
            scale = self.scale_act(x[..., 4:7])
            rotation = self.rot_act(x[..., 7:11])
            rgbs = self.rgb_act(x[..., 11:])


        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians
    
    def get_gaussian_loss(self, gt_gaussians, pred_gaussians):
        gs_loss = 0

        for b in range(gt_gaussians.shape[0]):
            gs_loss += F.mse_loss(gt_gaussians[b], pred_gaussians[b], reduction='mean') 

        return gs_loss / gt_gaussians.shape[0]
    
    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        images = data['input'] # [B, 4, 9, h, W], input features
        if self.opt.use_drag_encoding:
            drags_start = data['drags_start'] # [B, N, 3], start points of drags
            drags_end = data['drags_end'] # [B, N, 3], move vectors of drags 
        else:
            drags_start = None
            drags_end = None
        
        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(images, drags_start, drags_end) # [B, N, 14]

        # always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]
        
        results['gaussians'] = gaussians

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
        
        if self.opt.lambda_flow > 0 and self.opt.stage1:
            loss_flow = self.get_gaussian_loss(data['gt_gaussians'], gaussians)
            results['loss_flow'] = loss_flow
            loss = self.opt.lambda_flow * loss_flow
        
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
        

        return results, drags_start, drags_end
    