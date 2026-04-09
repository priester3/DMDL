from __future__ import absolute_import

import inspect
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn


RUNTIME_MODES = ("strict", "fast")
AMP_MODES = ("no-amp", "fp16", "bf16")


def seed_everything(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(args):
    runtime_mode = getattr(args, "runtime_mode", "fast")
    if runtime_mode not in RUNTIME_MODES:
        raise ValueError("Unsupported runtime mode: {0}".format(runtime_mode))

    if runtime_mode == "fast":
        cudnn.benchmark = True
        cudnn.deterministic = False
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    compile_cache_dir = getattr(args, "compile_cache_dir", None)
    if getattr(args, "use_compile", False) and compile_cache_dir:
        os.makedirs(compile_cache_dir, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", compile_cache_dir)


def build_loader_kwargs(args, drop_last=False, pin_memory=None):
    workers = getattr(args, "workers", 0)
    kwargs = {
        "num_workers": workers,
        "pin_memory": getattr(args, "pin_memory", True) if pin_memory is None else pin_memory,
        "drop_last": drop_last,
    }
    if workers > 0:
        kwargs["persistent_workers"] = getattr(args, "persistent_workers", True)
        prefetch_factor = getattr(args, "prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def move_to_cuda(value, args):
    if not torch.is_tensor(value) or not torch.cuda.is_available():
        return value

    value = value.cuda(non_blocking=getattr(args, "non_blocking", True))
    if getattr(args, "channels_last", False) and value.dim() == 4:
        value = value.contiguous(memory_format=torch.channels_last)
    return value


def get_inference_autocast_context(args):
    if args is None:
        return nullcontext()

    requested_mode = getattr(args, "amp_mode", "fp16")
    if requested_mode not in AMP_MODES:
        raise ValueError("Unsupported amp mode: {0}".format(requested_mode))

    if not torch.cuda.is_available() or requested_mode == "no-amp":
        return nullcontext()

    dtype = torch.float16
    if requested_mode == "bf16":
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_supported else torch.float16

    kwargs = {"enabled": True}
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        return torch.cuda.amp.autocast(**kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        return torch.cuda.amp.autocast(**kwargs)


def wrap_model_for_training(model, args):
    model = model.cuda()
    if getattr(args, "channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    model = nn.DataParallel(model)

    if getattr(args, "use_compile", False):
        if hasattr(torch, "compile"):
            compile_kwargs = {}
            compile_mode = getattr(args, "compile_mode", "default")
            if compile_mode not in (None, "", "default"):
                compile_kwargs["mode"] = compile_mode
            try:
                model = torch.compile(model, **compile_kwargs)
            except Exception as exc:
                print("=> torch.compile unavailable for current model/runtime, fallback to eager: {0}".format(exc))
        else:
            print("=> torch.compile requested but torch {0} does not support it; fallback to eager.".format(torch.__version__))
    return model


def build_optimizer(model, args):
    params = [value for _, value in model.named_parameters() if value.requires_grad]
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    if getattr(args, "use_fused_optimizer", False):
        if "fused" in inspect.signature(torch.optim.Adam).parameters:
            optimizer_kwargs["fused"] = True
        else:
            print("=> fused Adam requested but torch {0} does not support it; fallback to standard Adam.".format(torch.__version__))

    return torch.optim.Adam(params, **optimizer_kwargs)


def log_acceleration_config(args):
    fused_available = "fused" in inspect.signature(torch.optim.Adam).parameters
    compile_available = hasattr(torch, "compile")
    print("==========")
    print("Acceleration:")
    print("  runtime_mode={0}".format(getattr(args, "runtime_mode", "fast")))
    print("  amp_mode={0}".format(getattr(args, "amp_mode", "fp16")))
    print("  pin_memory={0}".format(getattr(args, "pin_memory", True)))
    print("  non_blocking={0}".format(getattr(args, "non_blocking", True)))
    print("  persistent_workers={0}".format(getattr(args, "persistent_workers", True)))
    print("  prefetch_factor={0}".format(getattr(args, "prefetch_factor", None)))
    print("  channels_last={0}".format(getattr(args, "channels_last", False)))
    print("  use_compile={0} (available={1}, mode={2})".format(
        getattr(args, "use_compile", False),
        compile_available,
        getattr(args, "compile_mode", "default"),
    ))
    print("  compile_cache_dir={0}".format(getattr(args, "compile_cache_dir", None)))
    print("  use_fused_optimizer={0} (available={1})".format(
        getattr(args, "use_fused_optimizer", False),
        fused_available,
    ))
    print("==========")


class AmpController(object):
    def __init__(self, args):
        requested_mode = getattr(args, "amp_mode", "fp16")
        if requested_mode not in AMP_MODES:
            raise ValueError("Unsupported amp mode: {0}".format(requested_mode))

        self.mode = requested_mode
        self.enabled = torch.cuda.is_available() and self.mode != "no-amp"
        self.dtype = None
        if self.mode == "fp16":
            self.dtype = torch.float16
        elif self.mode == "bf16":
            bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            if not torch.cuda.is_available() or not bf16_supported:
                print("=> bf16 requested but current CUDA runtime does not support it; fallback to fp16.")
                self.mode = "fp16"
                self.dtype = torch.float16
            else:
                self.dtype = torch.bfloat16

        scaler_enabled = self.enabled and self.mode == "fp16"
        self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    def autocast(self):
        if not self.enabled:
            return nullcontext()

        kwargs = {"enabled": True}
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype
        try:
            return torch.cuda.amp.autocast(**kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            return torch.cuda.amp.autocast(**kwargs)

    def step(self, loss, optimizer):
        optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()


def get_amp_controller(owner, args):
    controller = getattr(owner, "_amp_controller", None)
    requested_mode = getattr(args, "amp_mode", "fp16")
    if controller is None or getattr(controller, "requested_mode", requested_mode) != requested_mode:
        controller = AmpController(args)
        controller.requested_mode = requested_mode
        owner._amp_controller = controller
    return controller
