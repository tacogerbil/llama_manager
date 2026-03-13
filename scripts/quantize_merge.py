import os
import subprocess
from typing import Dict, Any, List
from .config import load_config
import glob




def list_models(cfg: Dict[str, Any]) -> List[str]:
    mp = cfg.get("model_path", "")
    if not mp:
        return []
    # Recursively find all .gguf files in subfolders
    pattern = os.path.join(mp, "**/*.gguf")
    matches = glob.glob(pattern, recursive=True)
    # Filter out projector files (mmproj-*.gguf, *-mmproj.gguf, *.mmproj)
    def is_projector_file(path):
        basename = os.path.basename(path)
        return (basename.startswith("mmproj-") and basename.endswith(".gguf")) or \
               basename.endswith("-mmproj.gguf") or \
               basename.endswith(".mmproj")
    # Return relative paths from model_path root to show folder structure, excluding projectors
    return sorted([os.path.relpath(p, mp) for p in matches if not is_projector_file(p)])




def quantize_model(cfg: Dict[str, Any], src_model: str, q_type: str, out_name: str | None = None):
    qbin = os.path.join(cfg["bin_path"], "llama-quantize")
    if os.name == "nt":
        qbin += ".exe"
    src_path = os.path.join(cfg["model_path"], src_model)
    if not out_name:
        out_name = src_model.replace('.gguf', f'-{q_type}.gguf')
    out_path = os.path.join(cfg["model_path"], out_name)
    cmd = [qbin, src_path, out_path, q_type]
    subprocess.run(cmd, check=True)




def find_split_files(cfg: Dict[str, Any]):
    mp = cfg.get("model_path", "")
    patterns = ["**/*-00001-of-*.gguf", "**/*.gguf.00001", "**/*-01-of-*.gguf", "**/*part1*.gguf"]
    matches = []
    for p in patterns:
        matches.extend(glob.glob(os.path.join(mp, p), recursive=True))
    return sorted(set(matches))




def merge_parts(cfg: Dict[str, Any], first_part: str, out_name: str):
    split_bin = os.path.join(cfg["bin_path"], "llama-gguf-split")
    if os.name == "nt":
        split_bin += ".exe"
    cmd = [split_bin, "--merge", first_part, os.path.join(cfg["model_path"], out_name)]
    subprocess.run(cmd, check=True)