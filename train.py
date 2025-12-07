import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SwinIRDataset
from model import SwinIR

def train():
    # --- Configuration ---
    # Running on Kaggle Tesla T4 (or locally for testing)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 5000
    BATCH_SIZE = 16 # Adjust based on VRAM (T4 has 16GB, usually 16-32 works with patch_size 48)
    PATCH_SIZE = 48
    LR_RATE = 2e-4
    CHECKPOINT_PATH = "checkpoint.pth"
    SAVE_INTERVAL = 500
    PRINT_INTERVAL = 10

    # Paths (Kaggle typically mounts data elsewhere, but using relative for this script as requested)
    HR_DIR = "data/train_hr"
    LR_DIR = "data/train_lr"

    # Ensure paths exist (fallback for case sensitivity)
    if not os.path.exists(HR_DIR): HR_DIR = 'data/train_HR'
    if not os.path.exists(LR_DIR): LR_DIR = 'data/train_LR'

    print(f"Starting training on {DEVICE}")
    print(f"HR: {HR_DIR}, LR: {LR_DIR}")
    print(f"Patch Size: {PATCH_SIZE}, Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")

    # --- Dataset & Dataloader ---
    # Note: debug_mode=False for production training
    train_dataset = SwinIRDataset(hr_dir=HR_DIR, lr_dir=LR_DIR, debug_mode=False, patch_size=PATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # --- Model ---
    # Using Lightweight configuration from model.py defaults
    model = SwinIR(depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], upscale=2)
    model = model.to(DEVICE)

    # --- Loss & Optimizer ---
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)

    start_epoch = 0

    # --- Persistence: Resume Training ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")

    # --- Training Loop ---
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            lr_imgs = data['LR'].to(DEVICE)
            hr_imgs = data['HR'].to(DEVICE)

            # Forward
            optimizer.zero_grad()
            outputs = model(lr_imgs)

            # Loss
            loss = criterion(outputs, hr_imgs)

            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Progress: Print every 10 epochs
        if (epoch + 1) % PRINT_INTERVAL == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

        # Saving: Save every 500 epochs
        # Also saving last epoch is good practice, but sticking strictly to prompt "Save ... every 500 epochs"
        if (epoch + 1) % SAVE_INTERVAL == 0:
            print(f"Saving checkpoint at epoch {epoch+1}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, CHECKPOINT_PATH)

    print("Training finished.")

    # Save final model
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, "final_model.pth")

if __name__ == "__main__":
    train()
