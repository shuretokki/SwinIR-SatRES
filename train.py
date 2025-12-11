"""
SwinIR Training Implementation
Based on: https://github.com/JingyunLiang/SwinIR

This script implements a training pipeline for SwinIR optimized for Satellite Imagery (xView),
utilizing generic Data Augmentation, AMP (Automatic Mixed Precision), and Tiled Processing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import kDataset
from model import SwinIR

from torch.cuda.amp import autocast, GradScaler
import math

def train(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    PATCH_SIZE = args.patch_size
    LR_RATE = 2e-4

    CHECKPOINT_PATH = "checkpoint.pth"
    SAVE_INTERVAL = 10
    PRINT_INTERVAL = 1

    HR_DIR = 'data/train_hr'
    LR_DIR = 'data/train_lr'

    if not os.path.exists(HR_DIR): HR_DIR = 'data/train_HR'
    if not os.path.exists(LR_DIR): LR_DIR = 'data/train_LR'

    print(f"[INFO] Starting training on {DEVICE}")
    print(f"Data - HR: {HR_DIR}, LR: {LR_DIR}")
    print(f"Config - Patch: {PATCH_SIZE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")

    train_dataset = kDataset(hr_dir=HR_DIR, lr_dir=LR_DIR, debug_mode=False, patch_size=PATCH_SIZE, upscale_factor=args.upscale)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    model = SwinIR(depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], upscale=args.upscale)
    model = model.to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Detected {torch.cuda.device_count()} GPUs. Enabling Multi-GPU")
        model = nn.DataParallel(model)

    criterion = nn.L1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=LR_RATE, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = GradScaler()

    start_epoch = 0

    # resume if crashed
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[INFO] Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Resuming from epoch {start_epoch}")
    else:
        print("[INFO] No checkpoint found. Starting from scratch.")


    avg_loss = 0.0
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            lr_imgs = data['LR'].to(DEVICE)
            hr_imgs = data['HR'].to(DEVICE)

            # forward with amp
            optimizer.zero_grad()

            with autocast():
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)

            # backward with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % PRINT_INTERVAL == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

        if (epoch + 1) % SAVE_INTERVAL == 0:
            print(f"[INFO] Saving checkpoint at epoch {epoch+1}...")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, CHECKPOINT_PATH)

    print("[INFO] Training finished.")

    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, "final_model.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--patch_size', type=int, default=48, help='Patch size (default: 48)')
    parser.add_argument('--epochs', type=int, default=5000, help='Total epochs (default: 5000)')
    parser.add_argument('--upscale', type=int, default=4, help='Upscale factor (default: 4)')
    args = parser.parse_args()

    train(args)
