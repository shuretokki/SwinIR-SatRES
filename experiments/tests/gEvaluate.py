import torch
import lpips
import numpy as np
import cv2
import os
from glob import glob
from skimage.metrics import structural_similarity as ssim

# Initialize LPIPS
loss_fn_alex = lpips.LPIPS(net='alex')

def load_image(path):
    # Read as BGR (cv2 default) then convert to RGB
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def to_tensor(img_np):
    # Normalize to [-1, 1] as per LPIPS requirement
    # img_np is 0-255
    img_norm = (img_np / 127.5) - 1.0
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0)
    return img_t

def calculate_psnr(img1, img2):
    # img1, img2: [0, 255] RGB
    # Convert to YCbCr and extract Y
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    # Shave border (scale=4)
    scale = 4
    img1 = img1[scale:-scale, scale:-scale]
    img2 = img2[scale:-scale, scale:-scale]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    # Convert to YCbCr and extract Y
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    scale = 4
    img1 = img1[scale:-scale, scale:-scale]
    img2 = img2[scale:-scale, scale:-scale]

    return ssim(img1, img2, data_range=255)

def main():
    base_dir = "data/test"
    gt_path = os.path.join(base_dir, "original.png")

    if not os.path.exists(gt_path):
        print("GT image not found!")
        return

    gt_img = load_image(gt_path)
    gt_tensor = to_tensor(gt_img)

    # Competitors
    models = [
        ("Bicubic", "bicubic.png"),
        ("SwinIR (Tiny)", "swin.png"),
        ("RealESRGAN", "realesrgan.png"),
        # ("HAT (Ref)", "hat_output.png"),  # Placeholders for: Chen et al. 2025
        # ("Liu et al. (Ref)", "liu_output.png") # Placeholders for: Liu et al. 2023
    ]

    print(f"{'Model':<20} | {'PSNR':<10} | {'SSIM':<10} | {'LPIPS':<10}")
    print("-" * 60)

    for name, filename in models:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"{name:<20} | Not Found")
            continue

        pred_img = load_image(path)

        # Ensure sizes match (crop if needed)
        h, w, c = gt_img.shape
        ph, pw, pc = pred_img.shape

        min_h = min(h, ph)
        min_w = min(w, pw)

        gt_crop = gt_img[:min_h, :min_w, :]
        pred_crop = pred_img[:min_h, :min_w, :]

        # PSNR/SSIM
        psnr_val = calculate_psnr(gt_crop, pred_crop)
        ssim_val = calculate_ssim(gt_crop, pred_crop)

        # LPIPS
        pred_tensor = to_tensor(pred_crop)
        gt_tensor_crop = to_tensor(gt_crop) # Re-tensorize cropped GT

        with torch.no_grad():
            lpips_val = loss_fn_alex(gt_tensor_crop, pred_tensor).item()

        print(f"{name:<20} | {psnr_val:<10.2f} | {ssim_val:<10.4f} | {lpips_val:<10.4f}")

if __name__ == "__main__":
    main()
