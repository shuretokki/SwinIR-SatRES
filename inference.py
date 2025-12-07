import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from model import SwinIR

def inference(image_path, model_path='final_model.pth', output_path='result.png'):
    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model Architecture
    # Must match training config: Lightweight SwinIR
    model = SwinIR(depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], upscale=2)
    model = model.to(device)

    # 2. Load Weights
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)

        # Handle state dict depending on how it was saved
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
    else:
        print(f"Error: Model file '{model_path}' not found. Train the model first!")
        return

    model.eval()

    # 3. Load & Preprocess Image
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    img = Image.open(image_path).convert('RGB')

    # Store original dimensions
    w, h = img.size
    print(f"Original image size: {w}x{h}")

    # Convert to Tensor
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device) # Add batch dimension -> [1, 3, H, W]

    # 4. Inference
    with torch.no_grad():
        output_tensor = model(img_tensor)

    # 5. Postprocess & Save
    # Remove batch dimension and clamp to [0, 1]
    output_tensor = output_tensor.squeeze(0).clamp(0, 1)

    to_pil = transforms.ToPILImage()
    output_img = to_pil(output_tensor.cpu())

    w_out, h_out = output_img.size
    print(f"Output image size: {w_out}x{h_out}")

    output_img.save(output_path)
    print(f"Super-resolved image saved to {output_path}")

if __name__ == "__main__":
    # Create a dummy LR image for testing if one doesn't exist
    test_img = "data/test_lr.png"
    if not os.path.exists(test_img):
        os.makedirs("data", exist_ok=True)
        # Create a small random image
        Image.new('RGB', (32, 32), color='blue').save(test_img)
        print(f"Created dummy test image at {test_img}")

    inference(test_img)
