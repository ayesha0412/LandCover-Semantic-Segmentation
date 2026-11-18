"""M0 environment check: Python, PyTorch, CUDA and RTX 5080 (sm_120) readiness.

Usage:
    python scripts/check_env.py --out outputs/m0

Runs small CUDA kernels in fp32, fp16-autocast and bf16-autocast, plus a
ResNet-50-sized convolution forward/backward, and records peak memory. It does
NOT estimate the training batch size; that is profiled in M2 with the real
model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path


def package_versions(names):
    """Return ``{name: version or None}`` for the given importable packages."""
    out = {}
    for n in names:
        try:
            m = importlib.import_module(n)
            out[n] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            out[n] = f"MISSING ({type(exc).__name__}: {exc})"
    return out


def cuda_smoke_test(torch):
    """Exercise the GPU with matmul/conv kernels under fp32 and AMP.

    Args:
        torch: The imported ``torch`` module.

    Returns:
        Dictionary of per-test status, max abs error versus a CPU/fp32
        reference where applicable, elapsed time and peak allocated memory.
    """
    dev = torch.device("cuda")
    res = {}
    torch.manual_seed(0)
    a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
    ref = a @ b
    gpu = (a.to(dev) @ b.to(dev)).cpu()
    res["matmul_fp32_max_abs_err"] = float((gpu - ref).abs().max())
    for dt in ("float16", "bfloat16"):
        with torch.autocast("cuda", dtype=getattr(torch, dt)):
            out = (a.to(dev) @ b.to(dev)).float().cpu()
        res[f"matmul_autocast_{dt}_rel_err"] = float((out - ref).norm() / ref.norm())
    conv = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 256, 3, padding=1, dilation=2),
    ).to(dev)
    x = torch.randn(2, 3, 512, 512, device=dev)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        y = conv(x)
    y.float().mean().backward()
    torch.cuda.synchronize()
    res["conv_fwd_bwd_amp_s"] = round(time.time() - t0, 4)
    res["conv_peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    res["conv_grad_finite"] = bool(torch.isfinite(conv[0].weight.grad).all())
    return res


def main() -> int:
    """Collect environment facts, run the GPU smoke test and write a JSON report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("outputs/m0"))
    args = ap.parse_args()
    rep = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(
            ["numpy", "cv2", "tifffile", "PIL", "pandas", "matplotlib", "yaml", "scipy",
             "torchvision", "segmentation_models_pytorch", "timm", "pytest"]
        ),
    }
    ok = True
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["torch_cuda_build"] = torch.version.cuda
        rep["cudnn"] = torch.backends.cudnn.version()
        rep["cuda_available"] = torch.cuda.is_available()
        rep["arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            cap = f"sm_{p.major}{p.minor}"
            free, total = torch.cuda.mem_get_info()
            rep.update(
                device=p.name,
                capability=cap,
                capability_in_build=cap in rep["arch_list"],
                total_mem_gb=round(total / 2**30, 2),
                free_mem_gb_at_start=round(free / 2**30, 2),
                bf16_supported=torch.cuda.is_bf16_supported(),
                multiprocessors=p.multi_processor_count,
            )
            rep["smoke"] = cuda_smoke_test(torch)
            ok = rep["capability_in_build"] and rep["smoke"]["matmul_fp32_max_abs_err"] < 1e-2 \
                and rep["smoke"]["conv_grad_finite"]
        else:
            ok = False
    except Exception as exc:
        rep["torch_error"] = f"{type(exc).__name__}: {exc}"
        ok = False
    rep["ok"] = ok
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "env_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
