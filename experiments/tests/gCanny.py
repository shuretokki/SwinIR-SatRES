import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_edge_map(image_path, title, ax):
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found.")
        return

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error reading {image_path}")
        return

    # Apply Canny Edge Detection
    # Thresholds (100, 200) are standard; we might adjust if too noisy
    edges = cv2.Canny(img, 100, 200)

    ax.imshow(edges, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

def main():
    base_dir = "data/test"
    output_path = "docs/figures/canny_comparison.png"
    os.makedirs("docs/figures", exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Ground Truth
    generate_edge_map(os.path.join(base_dir, "original.png"), "Ground Truth (Edges)", axes[0])

    # 2. Bicubic
    generate_edge_map(os.path.join(base_dir, "bicubic.png"), "Bicubic (Edges)", axes[1])

    # 3. Swin (Ours)
    generate_edge_map(os.path.join(base_dir, "swin.png"), "Ours (Tiny) (Edges)", axes[2])

    # 4. ESRGAN (Comparison if available)
    generate_edge_map(os.path.join(base_dir, "realesrgan.png"), "ESRGAN (Ref) (Edges)", axes[3])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved edge comparison to {output_path}")

if __name__ == "__main__":
    main()
