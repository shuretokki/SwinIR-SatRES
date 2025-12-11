# Optimizing Swin Transformer for Edge-Based Aerial Surveillance: An Empirical Study

### Abstract

We present an empirical study on the feasibility of running Vision Transformers on edge-based aerial surveillance platforms. While consumer-grade Unmanned Aerial Vehicles (UAVs) are widely used, their on-board sensors often yield noisy, low-resolution imagery. Deep learning-based Super-Resolution (SR) offers a solution, but state-of-the-art models are computationally prohibitive. In this work, we analyze the performance trade-offs of a "Tiny" Swin Transformer specifically constrained by a strict 4GB VRAM budget. We establish a baseline for transformer-based deployment on entry-level hardware. Utilizing the VisDrone dataset, we demonstrate that while our optimized model trails behind specialized CNN-based lightweight models (e.g., IMDN) in peak PSNR (-0.25 dB), it achieves a 4x inference speedup (120ms) compared to the standard SwinIR baseline. This work provides a critical feasibility study on the "cost" of Transformer inductive biases in highly constrained edge environments.

---



# Getting Started

## Prerequisites

* **Git** installed.
* **Python 3.9+** installed.
* **uv** installed (The extremely fast Python package installer).
  * *Pip:* `pip install uv`
  * *MacOS/Linux:* `curl -LsSf https://astral.sh/uv/install.sh | sh`
  * *Windows:* `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
* **Node.js 20+** and **npm** installed.
* A GPU (NVIDIA) is recommended for the API, but it will fallback to CPU if not available.

---

## 1. Clone the Repository

First, clone the project to your local machine:

```bash
git clone https://github.com/shuretokki/vdronerez-swinir.git
cd vdronerez-swinir
```

---

## 2. Setup and Run the API (Backend)

The backend is built with FastAPI. We use `uv` for blazing fast setup.

### 2.1. Navigate to the API Folder

```bash
cd api
```

### 2.2. Create Virtual Environment & Install Dependencies

Using `uv`, this happens instantly:

```bash
# Create a virtual environment at .venv
uv venv

# Activate the environment
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

> **Note:** If you have an NVIDIA GPU, ensure `onnxruntime-gpu` is installed. You can force it with:
> `uv pip install onnxruntime-gpu`

### 2.3. Run the Server

Start the API server using Uvicorn:

```bash
uvicorn main:app --reload
```

The API should now be running at `http://localhost:8000`.
You can verify it's working by visiting the docs at `http://localhost:8000/docs`.

---

## 3. Setup and Run the Web Interface (Frontend)

The frontend is a Vue 3 application that interacts with the API.

> **Important:** Open a **new terminal window** for these steps (keep the API running in the first one).

### 3.1. Navigate to the Web Folder

From the root of the project:

```bash
cd web
```

### 3.2. Install Dependencies

```bash
npm install
```

### 3.3. Run the Development Server

```bash
npm run dev
```

The command console will show you the local URL, usually:
`http://localhost:5173/`

---

## 4. Using the Application

1. Open your browser and go to `http://localhost:5173/`.
2. **Check Connection:** Ensure the "SwinIR API" status indicator in the bottom sidebar is green.
3. **Run Inference:**
   * Click **Run** or use the file input to upload a low-resolution image.
   * The browser will send the image to your local API (`localhost:8000`).
   * Wait for the processing to finish.
   * Compare the "Before" and "After" results using the slider.

## Troubleshooting

* **API Connection Error:** If the web app says "API Offline", make sure your API terminal is still running.
* **Missing Model:** If the API fails with "Model not found", ensure `experiments/models/vdroneswin.onnx` exists.

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
