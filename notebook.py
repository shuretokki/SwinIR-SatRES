import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # 📷 SwinIR Super-Resolution Studio

    Welcome to your interactive **SwinIR** workbench! Use this notebook to explore your dataset, test the model, and run super-resolution on new images.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. 📂 Dataset Explorer
    """)
    return


@app.cell
def _(mo):
    rerun_btn = mo.ui.button(label="🎲 Shuffle Sample")
    return (rerun_btn,)


@app.cell
def _(dataset, mo, os, random, rerun_btn, transforms):
    # Load dataset sample
    def get_sample_pair():
        hr_dir = "data/train_hr"
        lr_dir = "data/train_lr"

        if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
            return None, None

        # Reuse our Dataset class logic roughly or just direct load for simplicity visuals
        ds = dataset.SwinIRDataset(hr_dir, lr_dir, debug_mode=False)
        if len(ds) == 0:
            return None, None

        idx = random.randint(0, len(ds) - 1)
        sample = ds[idx]
        return sample["LR"], sample["HR"]


    # State validation
    # dependent on rerun_btn.value to trigger re-execution
    _ = rerun_btn.value
    sample_lr, sample_hr = get_sample_pair()

    content = ""
    if sample_lr is not None:
        # Convert tensors to PIL for display
        _to_pil = transforms.ToPILImage()
        img_lr = _to_pil(sample_lr)
        img_hr = _to_pil(sample_hr)

        content = mo.hstack(
            [
                mo.vstack([mo.md("**Low Resolution (LR)**"), mo.image(img_lr)]),
                mo.vstack([mo.md("**High Resolution (HR)**"), mo.image(img_hr)]),
            ],
            justify="center",
            gap=2,
        )
    else:
        content = mo.md(
            "⚠️ *No data found in `data/train_hr` or `data/train_lr`. Run `prepare_data.py` first!*"
        )

    mo.vstack([rerun_btn, content])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. ⚡ Model Inference Playground
    """)
    return


@app.cell
def _(mo):
    checkpoint_exists = mo.ui.checkbox(
        label="Check for trained model (final_model.pth)", value=True
    )
    return


@app.cell
def _(model, os, torch):
    # Load Model Logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status_msg = ""
    loaded_model = None

    if os.path.exists("final_model.pth"):
        try:
            # Lightweight config matches our train.py
            _m = model.SwinIR(
                depths=[6, 6, 6, 6],
                embed_dim=60,
                num_heads=[6, 6, 6, 6],
                upscale=2,
            )
            _ckpt = torch.load("final_model.pth", map_location=device)
            if "model_state_dict" in _ckpt:
                _m.load_state_dict(_ckpt["model_state_dict"])
            else:
                _m.load_state_dict(_ckpt)
            _m.to(device)
            _m.eval()
            loaded_model = _m
            status_msg = "✅ **Model Loaded Successfully!** (Lightweight 2x)"
        except Exception as e:
            status_msg = f"❌ Error loading model: {e}"
    else:
        status_msg = "⚠️ `final_model.pth` not found. Please train the model first."

    # Return model container to be used in next cells
    return loaded_model, status_msg


@app.cell
def _(mo, status_msg):
    mo.md(status_msg)
    return


@app.cell
def _(mo):
    # Upload User Image
    image_upload = mo.ui.file(kind="button", label="📤 Upload Image for SR")
    return (image_upload,)


@app.cell
def _(image_upload, loaded_model, mo, torch, transforms):
    result_view = mo.md("*Upload an image to see results...*")

    if image_upload.value and loaded_model:
        # Process Upload
        try:
            from PIL import Image
            import io

            # Read image
            uploaded_image = Image.open(
                io.BytesIO(image_upload.value[0].contents)
            ).convert("RGB")
            w, h = uploaded_image.size

            # Inference
            to_tensor = transforms.ToTensor()
            model_input = (
                to_tensor(uploaded_image)
                .unsqueeze(0)
                .to(loaded_model.conv_first.weight.device)
            )

            with torch.no_grad():
                out_tensor = loaded_model(model_input)

            out_tensor = out_tensor.squeeze(0).clamp(0, 1).cpu()
            to_pil = transforms.ToPILImage()
            result_img = to_pil(out_tensor)

            result_view = mo.vstack(
                [
                    mo.md(
                        f"**Original Size**: {w}x{h} → **Output Size**: {result_img.size[0]}x{result_img.size[1]}"
                    ),
                    mo.hstack(
                        [
                            mo.vstack(
                                [mo.md("**Input**"), mo.image(uploaded_image)]
                            ),
                            mo.vstack(
                                [mo.md("**SwinIR Output**"), mo.image(result_img)]
                            ),
                        ],
                        justify="center",
                        gap=2,
                    ),
                ]
            )

        except Exception as e:
            result_view = mo.md(f"❌ Error processing image: {e}")

    mo.vstack([image_upload, result_view])
    return


@app.cell
def _():
    import dataset
    import model
    import torch
    import os
    import random
    import marimo as mo
    from torchvision import transforms
    return dataset, mo, model, os, random, torch, transforms


if __name__ == "__main__":
    app.run()
