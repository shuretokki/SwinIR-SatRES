import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SwinIRDataset
from model import SwinIR

from torch.cuda.amp import autocast, GradScaler
import math

def train():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    EPOCHS = 5000
    BATCH_SIZE = 16
    PATCH_SIZE = 48
    LR_RATE = 2e-4

    CHECKPOINT_PATH = "checkpoint.pth"
    SAVE_INTERVAL = 500
    PRINT_INTERVAL = 10

    # paths (relative bc kaggle mount points r weird sometimes)
    HR_DIR = "data/train_hr"
    LR_DIR = "data/train_lr"

    if not os.path.exists(HR_DIR): HR_DIR = 'data/train_HR'
    if not os.path.exists(LR_DIR): LR_DIR = 'data/train_LR'

    print(f"[INFO] Starting training on {DEVICE}")
    print(f"Data - HR: {HR_DIR}, LR: {LR_DIR}")
    print(f"Config - Patch: {PATCH_SIZE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")

    # dataset setup
    # turn off debug mode for actual training run
    train_dataset = SwinIRDataset(hr_dir=HR_DIR, lr_dir=LR_DIR, debug_mode=False, patch_size=PATCH_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = SwinIR(depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], upscale=2)
    model = model.to(DEVICE)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    scaler = GradScaler() # for mixed precision


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
    train()
