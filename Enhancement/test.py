import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.UHDM_arch import UHDM
from basicsr.metrics import calculate_psnr, calculate_ssim, calculate_niqe, calculate_lpips
from skimage import img_as_ubyte
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def self_ensemble(x, s, model):
    def forward_transformed(x, s, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
            s = torch.flip(s, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
            s = torch.flip(s, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
            s = torch.rot90(s, dims=(-2, -1))
        x = model(x, s)[0]
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, s, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

parser = argparse.ArgumentParser(description='Image Enhancement with Metrics')
parser.add_argument('--input_dir', default='test/input', type=str)
parser.add_argument('--input_s_dir', default='test/input_s', type=str)
parser.add_argument('--gt_dir', default='test/target', type=str)
parser.add_argument('--result_dir', default='test_v1-results', type=str)
parser.add_argument('--weights', default='weights/net_g_227000.pth', type=str)
parser.add_argument('--dataset', default='default', type=str)
args = parser.parse_args()

# Load config
yaml_file = 'Options/Ntire25_LowLight.yml'
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
x['network_g'].pop('type')

# Load model
model_restoration = UHDM(**x['network_g'])
print('total parameters:', sum(p.numel() for p in model_restoration.parameters()))
checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'], strict=False)
print("===> Testing using weights:", args.weights)
model_restoration.cuda().eval()

# Prepare paths
result_dir = os.path.join(args.result_dir, os.path.basename(args.weights).split('.')[0])
os.makedirs(result_dir, exist_ok=True)
input_paths = natsorted(glob(os.path.join(args.input_dir, '*.png')))
input_s_paths = natsorted(glob(os.path.join(args.input_s_dir, '*.png')))
gt_paths = natsorted(glob(os.path.join(args.gt_dir, '*.png')))

# Metrics
psnr_list, ssim_list = [], []
niqe_list, lpips_list = [], []
metric_log_path = os.path.join(result_dir, 'metrics.csv')
with open(metric_log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'PSNR', 'SSIM', 'NIQE', 'LPIPS'])

    with torch.inference_mode():
        for inp_path, inp_s_path, gt_path in tqdm(zip(input_paths, input_s_paths, gt_paths), total=len(gt_paths)):
            img = np.float32(utils.load_img(inp_path)) / 255.
            img_s = np.float32(utils.load_img(inp_s_path)) / 255.
            gt = np.float32(utils.load_img(gt_path)) / 255.

            input_ = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
            input_s = torch.from_numpy(img_s).permute(2, 0, 1).unsqueeze(0).cuda()
            gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).cuda()

            # Pad if needed
            h, w = input_.shape[2:]
            H, W = ((h+31)//32)*32, ((w+31)//32)*32
            pad_h, pad_w = H-h, W-w
            input_ = F.pad(input_, (0, pad_w, 0, pad_h), 'reflect')
            input_s = F.pad(input_s, (0, pad_w, 0, pad_h), 'reflect')

            restored = self_ensemble(input_, input_s, model_restoration)[:,:,:h,:w]
            restored = torch.clamp(restored, 0, 1)

            # Save image
            out_img = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            utils.save_img(os.path.join(result_dir, os.path.basename(inp_path)), img_as_ubyte(out_img))

            # Metrics
            psnr_val = calculate_psnr(restored, gt_tensor, crop_border=0, test_y_channel=False)
            ssim_val = calculate_ssim(restored, gt_tensor, crop_border=0, test_y_channel=False)
            niqe_input = restored.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            niqe_val = calculate_niqe(niqe_input.astype(np.float32), crop_border=0)
            lpips_val = calculate_lpips(restored, gt_tensor)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            niqe_list.append(niqe_val)
            lpips_list.append(lpips_val)

            writer.writerow([
                os.path.basename(inp_path),
                f"{psnr_val:.4f}",
                f"{ssim_val:.4f}",
                f"{float(niqe_val):.4f}",
                f"{lpips_val:.4f}"
            ])

# Final log
print(f"\n✅ Saved metrics to {metric_log_path}")
print(f"📈 Average PSNR: {np.mean(psnr_list):.4f} dB")
print(f"📈 Average SSIM: {np.mean(ssim_list):.4f}")
print(f"📉 Average NIQE: {np.mean(niqe_list):.4f}")
print(f"🔁 Average LPIPS: {np.mean(lpips_list):.4f}")
