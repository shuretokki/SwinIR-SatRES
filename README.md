# SwinIR-SatRes: Satellite Imagery Super-Resolution

---

### Abstract

This repository presents **SwinIR-SatRes**, an adaptation of the state-of-the-art **SwinIR** (Image Restoration Using Swin Transformer) specifically engineered for satellite imagery enhancement. Unlike standard super-resolution tasks, satellite imagery presents unique challenges including massive file sizes (GeoTIFF), high-frequency texture details, and variable ground sampling distances (GSD).

---

### Usage

#### 1. Data Preparation

Processing massive satellite images (TIFF) into training-ready 512x512 chips.

```bash
python dataprep.py --source /path/to/xview/train_images --limit 500
```

#### 2. Training

Train the SwinIR model

```bash
python train.py --upscale 4 --batch_size 32 --patch_size 48 --epochs 500
```

*Supports Multi-GPU training automatically if available.*

#### 3. Inference (Super-Resolution)

Apply the trained model to new images.

```bash
python swin.py -i input.png -o output.png -m final_model.pth -s 4
```

---

### Results

*Results section to be updated after training completion.*

---

### Citation

If you use this code, please cite the original SwinIR paper:

```bibtex
@inproceedings{liang2021swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE/CVF International Conference on Computer Vision Workshops},
  pages={1833--1844},
  year={2021}
}
```

### License

This project is released under the Apache 2.0 license.
Based on [SwinIR](https://github.com/JingyunLiang/SwinIR).
