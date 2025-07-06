import torch
import lpips

loss_fn = lpips.LPIPS(net='alex').cuda()

def calculate_lpips(img1, img2):
    # Inputs must be in [-1, 1] range
    img1 = (img1 * 2) - 1
    img2 = (img2 * 2) - 1
    return loss_fn(img1, img2).item()
