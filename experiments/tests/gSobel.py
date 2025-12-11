import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

def apply_sobel(image_path, title, ax):
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found.")
        return

    # Load image and convert to grayscale tensor
    img = Image.open(image_path).convert('L')
    img_tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0) / 255.0

    # Define Sobel kernels
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

    # Apply convolution
    grad_x = F.conv2d(img_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(img_tensor, sobel_y, padding=1)

    # Magnitude
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    # Thresholding for cleaner edges (optional, imitating Canny)
    # Normalize to 0-1
    grad_magnitude = grad_magnitude / grad_magnitude.max()
    edges = (grad_magnitude > 0.15).float().squeeze().numpy()

    ax.imshow(edges, cmap='gray_r') # Inverted grayscale for better visibility (black edges on white)
    ax.set_title(title)
    ax.axis('off')

def main():
    base_dir = "data/test"
    output_path = "docs/figures/sobel_comparison.png"
    os.makedirs("docs/figures", exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    apply_sobel(os.path.join(base_dir, "original.png"), "Ground Truth (Edges)", axes[0])
    apply_sobel(os.path.join(base_dir, "bicubic.png"), "Bicubic (Edges)", axes[1])
    apply_sobel(os.path.join(base_dir, "swin.png"), "Ours (Tiny) (Edges)", axes[2])
    apply_sobel(os.path.join(base_dir, "realesrgan.png"), "ESRGAN (Ref) (Edges)", axes[3])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved edge comparison to {output_path}")

if __name__ == "__main__":
    main()
