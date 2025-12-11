# Optimizing Swin Transformer for Edge-Based Aerial Surveillance: An Empirical Study

### Abstract

We present an empirical study on the feasibility of running Vision Transformers on edge-based aerial surveillance platforms. While consumer-grade Unmanned Aerial Vehicles (UAVs) are widely used, their on-board sensors often yield noisy, low-resolution imagery. Deep learning-based Super-Resolution (SR) offers a solution, but state-of-the-art models are computationally prohibitive. In this work, we analyze the performance trade-offs of a "Tiny" Swin Transformer specifically constrained by a strict 4GB VRAM budget. We establish a baseline for transformer-based deployment on entry-level hardware. Utilizing the VisDrone dataset, we demonstrate that while our optimized model trails behind specialized CNN-based lightweight models (e.g., IMDN) in peak PSNR (-0.25 dB), it achieves a 4x inference speedup (120ms) compared to the standard SwinIR baseline. This work provides a critical feasibility study on the "cost" of Transformer inductive biases in highly constrained edge environments.

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
