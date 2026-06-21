"""Modified from https://github.com/kijai/ComfyUI-MochiWrapper
"""
import torch
import torch.nn as nn


def replace_parameters_by_name(module, name_keywords, device):
    for name, param in list(module.named_parameters(recurse=False)):
        if any(keyword in name for keyword in name_keywords):
            if isinstance(param, nn.Parameter):
                tensor = param.data
                delattr(module, name)
                setattr(module, name, tensor.to(device=device))
    for _, child_module in module.named_children():
        replace_parameters_by_name(child_module, name_keywords, device)


def convert_model_weight_to_float8(model, exclude_module_name=['embed_tokens']):
    for name, module in model.named_modules():
        if any(ex in name for ex in exclude_module_name):
            continue

        for param_name, param in module.named_parameters():
            if any(ex in param_name for ex in exclude_module_name):
                continue

            param.data = param.data.to(torch.float8_e4m3fn)


def convert_weight_dtype_wrapper(module, origin_dtype):
    for name, mod in module.named_modules():
        # skip root and embedding layers
        if name == "" or "embed_tokens" in name:
            continue

        # avoid wrapping twice
        if hasattr(mod, "original_forward"):
            continue

        if hasattr(mod, "weight") and mod.weight is not None:
            orig_forward = mod.forward

            # unwrap accelerate / decorator wrappers if present
            while hasattr(orig_forward, "__wrapped__"):
                orig_forward = orig_forward.__wrapped__

            mod.original_forward = orig_forward

            def new_forward(*inputs, m=mod, **kwargs):
                casted_inputs = [
                    inp.to(origin_dtype) if torch.is_tensor(inp) else inp
                    for inp in inputs
                ]
                return m.original_forward(*casted_inputs, **kwargs)

            mod.forward = new_forward