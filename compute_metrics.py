import os
import pytorch_msssim

from pytorch_msssim import ssim
import lpips
import torch

import cv2

from tqdm import tqdm

VAL_FILELIST = '/path/to/your/val/filelist'

# src_images: N*3*256*256
# tgt_images: N*3*256*256
# origin_images: N*3*256*256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_ssim(src_images, tgt_images):
    total_ssim_loss = 0
    for i in range(len(src_images)):
        src_images[i] = src_images[i].unsqueeze(0).detach()
        tgt_images[i] = tgt_images[i].unsqueeze(0).detach()

        ssim_val = ssim(src_images, tgt_images, data_range=1.0, size_average=True)
        total_ssim_loss += ssim_val
    
    total_ssim_loss /= len(src_images)
    return total_ssim_loss

def compute_lpips(src_images, tgt_images, lpips_loss_fn):
    total_lpips_loss = 0
    for i in range(len(src_images)):
        src_images[i] = src_images[i].unsqueeze(0).detach()
        tgt_images[i] = tgt_images[i].unsqueeze(0).detach()
        with torch.no_grad():
            lpips_loss_val = lpips_loss_fn(src_images, tgt_images).mean()
        total_lpips_loss += lpips_loss_val
        del lpips_loss_val
        torch.cuda.empty_cache()

    total_lpips_loss /= len(src_images)
    return total_lpips_loss.mean()

def compute_psnr(src_images, tgt_images):
    psnr_val = -10 * torch.log10(torch.mean((src_images - tgt_images) ** 2))
    return psnr_val

def process_image(image_path):
    image = cv2.imread(image_path)
    image = image[:, :2048, :]
    image = cv2.resize(image, (2048, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    return image

if __name__ == '__main__':
    count = 0

    total_lpips_loss = 0
    total_ssim_loss = 0
    total_psnr_loss = 0

    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    lpips_loss.eval()
    lpips_loss.requires_grad = False


    with open(VAL_FILELIST, 'r') as f:
        for line in tqdm(f.readlines()):
            tgt_path, src_path, origin_path = line.strip().split(',')
            src_image = process_image(src_path)
            tgt_image = process_image(tgt_path)
            origin_image = process_image(origin_path)
 
            for j in range(8):
                count += 1
                crop_src_image = src_image[:, :, j*256:(j+1)*256]
                crop_tgt_image = tgt_image[:, :, j*256:(j+1)*256]
                crop_origin_image = origin_image[:, :, j*256:(j+1)*256]
                
                psnr_val = compute_psnr(crop_src_image.unsqueeze(0), crop_tgt_image.unsqueeze(0))
                ssim_val = compute_ssim(crop_src_image.unsqueeze(0), crop_tgt_image.unsqueeze(0))
                lpips_loss_val = compute_lpips(crop_src_image.unsqueeze(0), crop_tgt_image.unsqueeze(0), lpips_loss)
                if torch.isinf(psnr_val):
                    print(f"psnr_val is inf, skip")
                    continue
                total_psnr_loss += psnr_val
                total_ssim_loss += ssim_val
                total_lpips_loss += lpips_loss_val

            del src_image, tgt_image, origin_image, psnr_val, ssim_val, lpips_loss_val


    psnr_val = total_psnr_loss / count
    ssim_val = total_ssim_loss / count
    lpips_loss_val = total_lpips_loss / count

    print("PSNR: ", psnr_val)
    print("SSIM: ", ssim_val)
    print("LPIPS: ", lpips_loss_val)


