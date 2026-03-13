import subprocess
import platform
import socket
import shlex
import json
from typing import Dict, Any, Optional
from .config import load_config


# Parse llama-server --help output for supported flags
def probe_binary(binary_path: str) -> Dict[str, bool]:
    """Return a dict of feature support as detected from the binary's --help output."""
    flags = {
        "flash_attn": False,
        "cache_ctk": False,
        "cache_ctv": False,
        "ctx_size_flag": False
    }
    try:
        out = subprocess.run([binary_path, "--help"], capture_output=True, text=True, timeout=5)
        help_text = out.stdout + out.stderr
    except Exception:
        help_text = ""

    h = help_text.lower()
    if any(k in h for k in ["flash-attn", "--flash-attention", "-fa"]):
        flags["flash_attn"] = True

    if any(k in h for k in ["-ctk", "--cache-type-k"]):
        flags["cache_ctk"] = True

    if any(k in h for k in ["-ctv", "--cache-type-v"]):
        flags["cache_ctv"] = True

    # context flag detection
    if any(k in h for k in ["--ctx-size", "--context"]):
        flags["ctx_size_flag"] = True



    return flags


# Simple model-family detection from filename. Can be extended to read GGUF metadata if present.
def detect_model_family(model_filename: str) -> str:
    n = model_filename.lower()
    # Check for vision models first (more specific)
    if "llava" in n:
        return "llava"
    if "minicpm" in n:
        return "minicpm"
    # Standard text models
    if "gemma" in n:
        return "gemma"
    if "mixtral" in n:
        return "mixtral"
    if "mistral" in n:
        return "mistral"
    if "qwen" in n:
        return "qwen"
    if "llama" in n or "vicuna" in n:
        return "llama"
    # fallback
    return "llama"


# Basic GPU probe using nvidia-smi if available
def probe_gpu_support(binary_path: str = None, debug: bool = False) -> Dict[str, bool]:
    """
    Detect GPU capabilities and supported llama-server flags.
    Returns a dict containing:
      - flash_attn, cache_ctk, cache_ctv, ctx_size_flag (from binary+GPU)
      - vendor, name, vram_gb, cuda, vulkan (GPU info)
    """
    flags = {
        "flash_attn": False,
        "cache_ctk": False,
        "cache_ctv": False,
        "ctx_size_flag": False,
        "vendor": None,
        "name": None,
        "vram_gb": 0,
        "cuda": False,
        "vulkan": False
    }

    # --- Binary flag probing ---
    help_text = ""
    if binary_path:
        try:
            out = subprocess.run([binary_path, "--help"], capture_output=True, text=True, timeout=5)
            help_text = (out.stdout + out.stderr).lower()
            if debug: print("Binary help output:\n", help_text)
        except Exception:
            if debug: print("Failed to run binary for detection.")

        if any(k in help_text for k in ["flash-attn", "--flash-attention", "-fa"]):
            flags["flash_attn"] = True
        if any(k in help_text for k in ["-ctk", "--cache-type-k"]):
            flags["cache_ctk"] = True
        if any(k in help_text for k in ["-ctv", "--cache-type-v"]):
            flags["cache_ctv"] = True
        if any(k in help_text for k in ["--ctx-size", "--context"]):
            flags["ctx_size_flag"] = True

    # --- GPU probing ---
    # NVIDIA detection
    try:
        if platform.system() in ["Windows", "Linux"]:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if out.returncode == 0 and out.stdout.strip():
                if out.returncode == 0 and out.stdout.strip():
                    lines = out.stdout.strip().split("\n")
                    gpus = []
                    total_vram = 0.0
                    
                    for i, line in enumerate(lines):
                        parts = line.split(",")
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            vram_mib = float(parts[1].strip())
                            vram_gb = vram_mib / 1024
                            gpus.append({
                                "index": i,
                                "name": name,
                                "vram_gb": vram_gb
                            })
                            total_vram += vram_gb
                    
                    if gpus:
                        # Use first GPU as primary for backward compatibility
                        flags["vendor"] = "NVIDIA"
                        flags["name"] = gpus[0]["name"]
                        flags["vram_gb"] = gpus[0]["vram_gb"] # Primary only for now, 'total_vram' available in 'gpus' list if needed
                        flags["cuda"] = True
                        flags["gpus"] = gpus # New field with all GPUs
                        
                        if debug: 
                            print(f"NVIDIA GPUs detected: {len(gpus)}")
                            for g in gpus:
                                print(f" - GPU {g['index']}: {g['name']} ({g['vram_gb']:.2f} GB)")
                        
                        return flags
    except Exception as e:
        if debug: print(f"NVIDIA detection failed: {e}")

    # AMD / ROCm detection
    try:
        if platform.system() == "Linux":
            out = subprocess.run(
                shlex.split("rocm-smi --showmeminfo vram --json"),
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0 and out.stdout.strip():
                vram_data = json.loads(out.stdout)
                # Find the first card and get its VRAM
                for card, info in vram_data.items():
                    if "VRAM Total Memory (B)" in info:
                        vram_bytes = int(info["VRAM Total Memory (B)"])
                        flags["vendor"] = "AMD"
                        flags["name"] = f"AMD ROCm GPU ({card})"
                        flags["vram_gb"] = vram_bytes / (1024**3)
                        flags["vulkan"] = True
                        if debug: print(f"AMD GPU detected: {flags['name']} {flags['vram_gb']:.2f} GB VRAM")
                        return flags
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        if debug: print(f"AMD ROCm detection failed: {e}")

    # Fallback for other GPUs (like Intel Arc) or if other methods fail
    if not flags["vendor"]:
        try:
            # This is a very basic check and won't give VRAM, but confirms presence
            out = subprocess.run(["vulkaninfo"], capture_output=True, text=True)
            if out.returncode == 0:
                flags["vendor"] = "Unknown (Vulkan compatible)"
                flags["vulkan"] = True
                if debug: print("Vulkan-compatible  GPU detected (vendor unknown).")
        except Exception:
            if debug: print("Vulkan detection failed.")

    # --- Feature sanity check based on GPU ---
    # Disable GPU-dependent features if no CUDA/Vulkan
    if not (flags["cuda"] or flags["vulkan"]):
        flags["flash_attn"] = False
        flags["cache_ctk"] = False
        flags["cache_ctv"] = False
        if debug: print("GPU not capable: disabled flash_attn/cache_ctk/cache_ctv")

    return flags