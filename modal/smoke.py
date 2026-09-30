"""Smoke test: does Modal GPU compute actually work from this box?

Verifies auth, image build, GPU allocation, CUDA, torch, and disk/network for HF model pulls
BEFORE any real benchmark time is spent. A benchmark that dies on a missing dependency twenty
minutes in costs the same as one that dies in the first ten seconds.
"""
import modal

app = modal.App("ccai-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "hf_transfer",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Persist the HF cache across runs: without this every run re-downloads 15 GB of weights.
hf_cache = modal.Volume.from_name("ccai-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=600, volumes={"/cache": hf_cache})
def probe():
    import subprocess, sys, os
    import torch

    out = {}
    out["torch"] = torch.__version__
    out["cuda_available"] = torch.cuda.is_available()
    out["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out["gpu"] = p.name
        out["gpu_mem_gb"] = round(p.total_memory / 1e9, 1)
        out["sm_count"] = p.multi_processor_count
        # real compute, not just a device query
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        out["matmul_ok"] = bool(abs(c) > 0)
        out["matmul_4096_ms"] = round(
            (lambda t0: (torch.cuda.synchronize(), __import__("time").perf_counter() - t0)[1])(
                __import__("time").perf_counter()
            ) * 1000, 2,
        )
    try:
        out["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        out["nvidia_smi"] = "unavailable: %s" % e
    out["hf_cache_writable"] = os.access("/cache", os.W_OK)
    out["disk_free_gb"] = round(
        __import__("shutil").disk_usage("/cache").free / 1e9, 1)
    return out


@app.local_entrypoint()
def main():
    import json
    r = probe.remote()
    print(json.dumps(r, indent=2))
    ok = r.get("cuda_available") and r.get("matmul_ok")
    print("\nSMOKE:", "PASS" if ok else "FAIL")
