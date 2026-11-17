"""fp16 backward headroom for the early decoder layers.

Measured on V100: the gradient flowing into decoder layers 0-4 is 10-100x larger than anywhere else in the
network (attention-sink token), so it is the first thing to overflow fp16 and it pins the GradScaler at a tiny
loss scale. `install_grad_downscale` inserts an identity op after layer `layer_idx` whose backward divides the
gradient by `factor`. Adam updates are invariant to a per-parameter constant gradient scale (m / sqrt(v)), so the
LoRA/embedding updates of layers 0..layer_idx are unchanged while the overflow threshold rises by `factor`.
"""
import torch


class _ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def install_grad_downscale(hf_model, layer_idx: int, factor: float):
    """hf_model: the AnoleforConditionalGeneration (unwrapped). Returns the hook handle."""
    layers = hf_model.model.layers
    scale = 1.0 / float(factor)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (_ScaleGrad.apply(output[0], scale),) + tuple(output[1:])
        return _ScaleGrad.apply(output, scale)

    return layers[layer_idx].register_forward_hook(hook)
