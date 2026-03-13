from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from typing import Dict, Any
import json
from pathlib import Path

console = Console()

def read_gguf_metadata(gguf_path: str) -> dict:
    """
    Read GGUF file metadata to extract model family, size, and tokenizer info.
    Returns a dict with keys like 'model_family', 'n_layers', 'd_model', 'tokenizer'.
    """
    metadata = {}
    gguf_file = Path(gguf_path)
    if not gguf_file.exists():
        return metadata

    try:
        with open(gguf_file, "rb") as f:
            header = f.read(16)  # GGUF magic + version
            # TODO: expand: parse metadata according to GGUF spec
            # For now, try to extract embedded JSON chunk if available
            f.seek(0)
            content = f.read()
            # naive search for JSON metadata
            start = content.find(b'{')
            end = content.rfind(b'}')
            if start != -1 and end != -1:
                json_bytes = content[start:end+1]
                metadata = json.loads(json_bytes.decode('utf-8', errors='ignore'))
    except Exception:
        pass

    return metadata


def render_model_info(model_name: str, family: str, capabilities: Dict[str, Any], recommended: Dict[str, Any]):
    t = Table.grid(expand=False)
    t.add_column(justify="right", style="bold")
    t.add_column()
    t.add_row("Model:", model_name)
    t.add_row("Family:", family)
    t.add_row("Flash Attention (binary+model+gpu):", "✅" if capabilities.get("flash_ok") else "❌")
    t.add_row("Cache Compression:", "✅" if capabilities.get("cache_ok") else "❌")
    t.add_row("Recommended NGL:", recommended.get("ngl", "99"))
    t.add_row("Notes:", recommended.get("note", ""))
    # recommended settings
    rec_lines = [f"ctx_size: {recommended.get('ctx_size')}", f"cache_type: {recommended.get('cache_type')}", f"gpu_layers: {recommended.get('gpu_layers')}"]
    t.add_row("Recommended:", "\n".join(rec_lines))
    panel = Panel(t, title="Model Info")
    console.print(panel)