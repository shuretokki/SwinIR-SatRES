import os
import glob
from PIL import Image

def prepare_dataset(hr_sourcedir='data/train'):
    """
    Prepares dataset by fixing HR images to be multiples of 4 and generating 4x downscaled LR images.
    """
    # Handle path case sensitivity or alternatives
    if not os.path.exists(hr_sourcedir) and os.path.exists('data/train'):
        hr_sourcedir = 'data/train'

    # Define output directories
    # Using a fixed folder prevents overwriting original data which is safer
    hr_fixed_dir = 'data/train_hr'
    lr_dir = 'data/train_lr'

    os.makedirs(hr_fixed_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    # Find images
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.BMP']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(hr_sourcedir, ext)))
        # Case insensitive check (e.g. .PNG) could be added but glob is usually case sensitive on Linux
        files.extend(glob.glob(os.path.join(hr_sourcedir, ext.upper())))

    files = sorted(list(set(files))) # Remove duplicates if any

    print(f"Found {len(files)} images in {hr_sourcedir}")

    count = 0
    last_hr_shape = (0, 0)
    last_lr_shape = (0, 0)

    for file_path in files:
        filename = os.path.basename(file_path)

        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                w, h = img.size

                # 1. Resize/Crop to multiple of 4
                w_new = w - (w % 4)
                h_new = h - (h % 4)

                if w_new != w or h_new != h:
                    # Crop center or top-left? Usually top-left is standard for simple robust cropping
                    # but center crop is nicer. Let's do simple slice for determinism and speed
                    img_fixed = img.crop((0, 0, w_new, h_new))
                else:
                    img_fixed = img

                # Save fixed HR
                img_fixed.save(os.path.join(hr_fixed_dir, filename))
                last_hr_shape = img_fixed.size

                # 2. Downscale by 4x
                # BICUBIC is standard for SR degradation model
                lr_w = w_new // 4
                lr_h = h_new // 4

                img_lr = img_fixed.resize((lr_w, lr_h), Image.BICUBIC)

                # Save LR
                img_lr.save(os.path.join(lr_dir, filename))
                last_lr_shape = img_lr.size

                count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processed {count} images.")
    if count > 0:
        print(f"Last processed pair - HR shape: {last_hr_shape}, LR shape: {last_lr_shape}")

if __name__ == "__main__":
    prepare_dataset()
