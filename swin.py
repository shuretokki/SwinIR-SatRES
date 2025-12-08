"""
SwinIR CLI Tool for Image Super-Resolution.

This script provides a Command Line Interface (CLI) to run SwinIR inference on images.
It supports tiled processing to handle large images on limited VRAM and automatic mixed precision.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import math
from model import SwinIR

import argparse

def inference(args):
    """
    Runs SwinIR inference on a single image.

    Args:
        args (Namespace): Parsed command-line arguments containing:
                          - input: Path to input image.
                          - output: Path to save output image.
                          - model: Path to model checkpoint.
                          - scale: Upscale factor.
                          - tile_size: Tile size for sliding window inference.
    """
    tile_size = args.tile_size
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] Detected GPU VRAM: {vram:.2f} GB")
        torch.backends.cudnn.benchmark = True

        if vram < 4.0 and tile_size > 256:
            print("[WARN] Low VRAM. Reducing tile size to 256.")
            tile_size = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # note: depths/embed_dim must match training. scale is configurable.
    print(f"[INFO] Initializing SwinIR model (Scale: {args.scale}x)...")
    model = SwinIR(depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], upscale=args.scale)
    model = model.to(device)

    if device.type == 'cuda':
        model.half()

    if os.path.exists(args.model):
        print(f"[INFO] Loading weights from {args.model}...")
        checkpoint = torch.load(args.model, map_location=device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        print(f"[ERROR] Model file '{args.model}' not found. Please train first.")
        return

    model.eval()

    if not os.path.exists(args.input):
        print(f"[ERROR] Image '{args.input}' not found.")
        return

    img = Image.open(args.input).convert('RGB')
    w, h = img.size
    print(f"Original Size: {w}x{h}")

    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device) # [1, 3, H, W]

    if device.type == 'cuda':
        img_tensor = img_tensor.half()

    b, c, h, w = img_tensor.shape
    tile = min(tile_size, h, w)
    tile_overlap = 32
    stride = tile - tile_overlap

    sf = args.scale

    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]

    E = torch.zeros(b, c, h*sf, w*sf).type_as(img_tensor)
    W = torch.zeros_like(E)

    print(f"[INFO] Processing {len(h_idx_list) * len(w_idx_list)} tiles...")

    window_size = 8

    with torch.no_grad():
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]

                # Auto-pad to multiple of window_size (8)
                _, _, h_old, w_old = in_patch.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old if h_old % window_size != 0 else 0
                w_pad = (w_old // window_size + 1) * window_size - w_old if w_old % window_size != 0 else 0

                if h_pad > 0 or w_pad > 0:
                     in_patch = torch.cat([in_patch, torch.flip(in_patch, [2])], 2)[:, :, :h_old + h_pad, :]
                     in_patch = torch.cat([in_patch, torch.flip(in_patch, [3])], 3)[:, :, :, :w_old + w_pad]

                out_patch = model(in_patch)

                # Crop back to original size * scale
                if h_pad > 0 or w_pad > 0:
                     out_patch = out_patch[..., :h_old*args.scale, :w_old*args.scale]

                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)

    output_tensor = E.div_(W)
    output_tensor = output_tensor.squeeze(0).clamp(0, 1).float()

    to_pil = transforms.ToPILImage()
    print("[INFO] Processing finished, saving image...")
    output_img = to_pil(output_tensor.cpu())

    w_out, h_out = output_img.size
    print(f"Output Size: {w_out}x{h_out}")

    output_img.save(args.output)
    print(f"[INFO] Saved result to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SwinIR Super-Resolution CLI')

    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input image')
    parser.add_argument('-o', '--output', type=str, default='result.png', help='Path to save output image')
    parser.add_argument('-m', '--model', type=str, default='final_model.pth', help='Path to model checkpoint')
    parser.add_argument('-s', '--scale', type=int, default=4, help='Upscale factor (default: 4)')
    parser.add_argument('-t', '--tile_size', type=int, default=512, help='Tile size for inference (default: 512)')

    args = parser.parse_args()

    inference(args)
