from model1 import HATIQCMix
import torch
from fvcore.nn import FlopCountAnalysis

def get_acts(model, input_tensor):
    total_acts = 0
    def hook(module, input, output):
        nonlocal total_acts
        if isinstance(output, torch.Tensor):
            total_acts += output.numel()

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)): # Acts usually counts output of convs/linear
            hooks.append(layer.register_forward_hook(hook))

    model(input_tensor)
    for h in hooks:
        h.remove()
    return total_acts

# Setup for 1280x720 Output (x4 scale -> 320x180 Input)
H, W = 180, 320
img_size = H # approximate patch size arg for model init if needed, though model usually adaptable

net = HATIQCMix(
    img_size=64, # keeping init arg, but forward will use real size
    patch_size=1,
    in_chans=3,
    embed_dim=48,
    depths=(6, 6, 6, 6),
    num_heads=(6, 6, 6, 6),
    window_size=16,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    mlp_ratio=2.,
    qkv_bias=True,
    upscale=4
)

input_tensor = torch.randn(1, 3, H, W)
flops = FlopCountAnalysis(net, input_tensor)
total_flops = flops.total()
total_prms = sum(p.numel() for p in net.parameters())
total_acts = get_acts(net, input_tensor)

print(f"\nResults for {W}x{H} Input -> {W*4}x{H*4} Output:")
print(f"Params: {total_prms / 1e6:.2f}M")
print(f"FLOPs: {total_flops / 1e9:.2f}G")
print(f"Acts: {total_acts / 1e6:.2f}M")
