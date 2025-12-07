import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import math
from model import SwinIR

def inference(image_path, model_path='final_model.pth', output_path='result.png', tile_size=512, tile_overlap=32):
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] Detected GPU VRAM: {vram:.2f} GB")
        torch.backends.cudnn.benchmark = True

        if vram < 4.0:
            print("[WARN] Low VRAM. Tile size set to 256.")
            tile_size = 256


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    model = SwinIR(depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], upscale=2)
    model = model.to(device)

    if device.type == 'cuda':
        model.half()


    if os.path.exists(model_path):
        print(f"[INFO] Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
    else:
        print(f"[ERROR] Model file '{model_path}' not found. Please train the model first.")
        return

    model.eval()

    if not os.path.exists(image_path):
        print(f"[ERROR] Image '{image_path}' not found.")
        return

    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    print(f"Original Size: {w}x{h}")

    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device) # [1, 3, H, W]

    if device.type == 'cuda':
        img_tensor = img_tensor.half()


    b, c, h, w = img_tensor.shape
    tile = min(tile_size, h, w)
    stride = tile - tile_overlap

    # scale factor
    sf = 2

    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]

    E = torch.zeros(b, c, h*sf, w*sf).type_as(img_tensor)
    W = torch.zeros_like(E)

    print(f"[INFO] Processing {len(h_idx_list) * len(w_idx_list)} tiles...")

    with torch.no_grad():
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:

                in_patch = img_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
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

    output_img.save(output_path)
    print(f"[INFO] Saved result to {output_path}")

if __name__ == "__main__":
    test_img = "data/test_lr.png"
    if not os.path.exists(test_img):
        print(f"[WARN] {test_img} not found, generating dummy...")
        os.makedirs("data", exist_ok=True)
        Image.new('RGB', (32, 32), color='blue').save(test_img)

    inference(test_img)
