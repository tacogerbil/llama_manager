import os
import json
import socket
from typing import Dict, Any
from pathlib import Path


CACHE_DIR = Path(__file__).parent.parent / "config"
CACHE_DIR.mkdir(exist_ok=True)


def get_config_path(hostname: str = None) -> Path:
    if hostname is None:
        hostname = socket.gethostname()
    return CACHE_DIR / f"{hostname}.json"


def get_flags_path(hostname: str = None) -> Path:
    if hostname is None:
        hostname = socket.gethostname()
    return CACHE_DIR / f"{hostname}_flags.json"


DEFAULT_CONFIG: Dict[str, Any] = {
    "bin_path": "",
    "model_path": "",
    "defaults": {
        "ctx_size": 8192,
        "cache_type": "q8_0",
        "gpu_layers": "99",
        "flash_attention": True,
        "kv_compression": True,
        "threads": None,
        "host": "0.0.0.0",
        "port": "8080",
        "api_key": "sk-37e0541fd0394cd7bc43fcbe89f8d8f6"
    },
    "model_defaults": {}
}


def cache_binary_flags(binary_path: str):
    from .detectors import probe_binary
    hostname = socket.gethostname()
    cache_file = get_flags_path(hostname)
    if cache_file.exists():
        try:
            return json.load(open(cache_file))
        except Exception:
            pass

    flags = probe_binary(binary_path)
    json.dump(flags, open(cache_file, "w"), indent=4)
    return flags


def load_config(path: Path | None = None) -> Dict[str, Any]:
    if path is None:
        path = get_config_path()
    if path.exists():
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = DEFAULT_CONFIG.copy()
    else:
        cfg = DEFAULT_CONFIG.copy()
    # ensure keys
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def save_config(cfg: Dict[str, Any], path: Path | None = None) -> None:
    if path is None:
        path = get_config_path()
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)