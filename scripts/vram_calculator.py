import os
import re
from typing import Dict, Any, Tuple, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# https://github.com/ggerganov/llama.cpp/blob/master/ggml.c
BPW_MAPPING = {
    'F32': 32.0, 'F16': 16.0, 'Q8_0': 8.5, 'Q8_1': 8.998,
    'Q6_K': 6.56,
    'Q5_0': 5.5, 'Q5_1': 5.625, 'Q5_K_S': 5.0625, 'Q5_K_M': 5.155,
    'Q4_0': 4.5, 'Q4_1': 4.7, 'Q4_K_S': 4.06, 'Q4_K_M': 4.43,
    'Q3_K_S': 3.06, 'Q3_K_M': 3.43, 'Q3_K_L': 3.81,
    'Q2_K': 2.56
}

_BYTES_PER_ELEM = {
    "f32": BPW_MAPPING['F32'] / 8.0,
    "f16": BPW_MAPPING['F16'] / 8.0,
    "q8_0": BPW_MAPPING['Q8_0'] / 8.0,
    "q8_1": BPW_MAPPING['Q8_1'] / 8.0,
    "q6_k": BPW_MAPPING['Q6_K'] / 8.0,
    "q5_0": BPW_MAPPING['Q5_0'] / 8.0,
    "q5_1": BPW_MAPPING['Q5_1'] / 8.0,
    "q5_k_s": BPW_MAPPING['Q5_K_S'] / 8.0,
    "q5_k_m": BPW_MAPPING['Q5_K_M'] / 8.0,
    "q4_0": BPW_MAPPING['Q4_0'] / 8.0,
    "q4_1": BPW_MAPPING['Q4_1'] / 8.0,
    "q4_k_s": BPW_MAPPING['Q4_K_S'] / 8.0,
    "q4_k_m": BPW_MAPPING['Q4_K_M'] / 8.0,
    "q3_k_s": BPW_MAPPING['Q3_K_S'] / 8.0,
    "q3_k_m": BPW_MAPPING['Q3_K_M'] / 8.0,
    "q3_k_l": BPW_MAPPING['Q3_K_L'] / 8.0,
    "q2_k": BPW_MAPPING['Q2_K'] / 8.0,
}

def get_params_from_filename(filename: str) -> float:
    """Extracts the number of billions of parameters from a model filename."""
    match = re.search(r'(\d+\.?\d*)[bB]', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0

def get_quant_from_filename(filename: str) -> str:
    """Extracts the quantization method from a model filename."""
    filename_upper = filename.upper().replace('.GGUF', '')
    for q_type in sorted(BPW_MAPPING.keys(), key=len, reverse=True):
        if q_type.replace('_', '') in filename_upper.replace('_', ''):
            return q_type
    return "Unknown"

def calculate_kv_cache_size(
    ctx_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    cache_type: str = "f16"
) -> float:
    """Calculates the VRAM required for the K-V cache using GQA formula."""
    bytes_per_elem = _BYTES_PER_ELEM.get(cache_type.lower(), 2.0)

    if num_heads == 0 or num_kv_heads == 0:
        return 0.0

    # The size of the K and V caches is determined by the number of KV heads,
    # not the full attention heads.
    # Formula: layers * context * (kv_heads * head_dim) * 2 (for K and V) * bytes_per_elem
    
    head_dim = hidden_size / num_heads
    
    total_size_bytes = num_layers * ctx_size * num_kv_heads * head_dim * 2 * bytes_per_elem
    
    return total_size_bytes / (1024**3)

def calculate_vram_usage(model_path: str, settings: Dict[str, Any],
                        arch_info: Dict[str, Any]) -> Dict[str, float]:
    """Calculate detailed VRAM breakdown using model parameters and heuristics."""
    model_filename = Path(model_path).name
    params_b = get_params_from_filename(model_filename)
    quant_type = get_quant_from_filename(model_filename)

    model_size_gb = 0
    if params_b > 0 and quant_type in BPW_MAPPING:
        bits_per_weight = BPW_MAPPING[quant_type]
        model_size_gb = (params_b * 1e9 * bits_per_weight) / (8 * 1024**3)
    else:
        try:
            model_size_gb = Path(model_path).stat().st_size / (1024**3)
        except:
            model_size_gb = 8.0

    gpu_layers = settings["gpu_layers"]
    total_layers = arch_info.get('n_layers', 1) or 1

    if gpu_layers == -1 or gpu_layers >= total_layers:
        model_weights_gb = model_size_gb
    elif gpu_layers == 0:
        model_weights_gb = 0
    else:
        # A more refined approximation for partial offloading
        model_weights_gb = model_size_gb * (gpu_layers / total_layers)

    # KV Cache calculation
    num_layers_for_kv = total_layers if gpu_layers >= total_layers else gpu_layers
    if gpu_layers == -1: # Another way to say 'all'
        num_layers_for_kv = total_layers

    kv_cache_gb = calculate_kv_cache_size(
        ctx_size=settings["ctx_size"],
        num_layers=num_layers_for_kv,
        hidden_size=arch_info.get('hidden_size', 4096),
        num_heads=arch_info.get('n_heads', 32),
        num_kv_heads=arch_info.get('n_kv_heads', arch_info.get('n_heads', 32)),
        cache_type=settings.get("cache_type", "f16")
    )

    # Projector weights for Vision models
    projector_gb = 0.0
    projector_path = settings.get("projector_path")
    if projector_path and os.path.exists(projector_path):
        try:
            projector_gb = Path(projector_path).stat().st_size / (1024**3)
        except:
            projector_gb = 0.0

    overhead_gb = 0.5

    raw_total = model_weights_gb + kv_cache_gb + projector_gb + overhead_gb

    safety_buffer = raw_total * 0.05
    total = raw_total + safety_buffer

    return {
        "model_weights": model_weights_gb,
        "kv_cache": kv_cache_gb,
        "projector_weights": projector_gb,
        "overhead": overhead_gb,
        "safety_buffer": safety_buffer,
        "raw_total": raw_total,
        "total": total,
        "total_layers": total_layers,
        "n_heads": arch_info.get('n_heads', 32),
        "n_kv_heads": arch_info.get('n_kv_heads', arch_info.get('n_heads', 32)),
        "hidden_size": arch_info.get('hidden_size', 0)
    }

def get_vram_available(vram_total_gb: float) -> float:
    """
    Calculates actually available VRAM.
    Subtracts typical Windows/driver overhead (varies by system).
    """
    # Conservative estimate: 300-500 MB overhead for WDDM + desktop
    overhead_gb = 0.4
    return max(0, vram_total_gb - overhead_gb)

def render_vram_display(model_name: str, model_path: str, vram_total: float,
                       settings: Dict[str, Any], safe_defaults: Dict[str, Any],
                       arch_info: Dict[str, Any], capabilities: Dict[str, Any] = None) -> None:
    """Render the interactive VRAM calculator display."""
    os.system('cls' if os.name == 'nt' else 'clear')

    usage = calculate_vram_usage(model_path, settings, arch_info)
    vram_available = get_vram_available(vram_total)
    vram_free = vram_available - usage["total"]
    fits = usage["total"] <= vram_available

    # Header with architecture info
    arch_name = arch_info.get('architecture', 'unknown')
    gqa_info = ""
    if usage.get('n_kv_heads') and usage.get('n_heads') and usage['n_kv_heads'] < usage['n_heads']:
        gqa_info = f" (GQA: {usage['n_kv_heads']})"

    console.print(Panel(
        f"[bold cyan]Interactive VRAM Configuration Tool[/bold cyan]\\n"
        f"Model: {model_name}\\n"
        f"Architecture: {arch_name} | {usage.get('total_layers', 'N/A')} layers | {usage.get('hidden_size', 'N/A')} hidden{gqa_info}\\n"
        f"GPU VRAM: {vram_total:.2f} GB Total | {vram_available:.2f} GB Available (GPUs: {settings.get('selected_gpus', '0')})",
        style="cyan"
    ))

    # Settings table
    settings_table = Table(show_header=False, box=None, padding=(0, 2))
    settings_table.add_column("Setting", style="bold")
    settings_table.add_column("Value")
    settings_table.add_column("Info")

    # Context size
    ctx_custom = settings["ctx_size"] != safe_defaults["ctx_size"]
    ctx_marker = " (custom)" if ctx_custom else ""
    max_ctx = arch_info.get('context_length', 32768)
    settings_table.add_row(
        "[C] Context Size:",
        f"{settings['ctx_size']}{ctx_marker}",
        f"[dim]safe: {safe_defaults['ctx_size']}, max: {max_ctx}[/dim]"
    )

    # GPU layers
    ngl_custom = settings["gpu_layers"] != safe_defaults["gpu_layers"]
    ngl_marker = " (custom)" if ngl_custom else ""
    ngl_display = "all" if settings.get('gpu_layers', 0) >= arch_info.get('n_layers', 999) else str(settings['gpu_layers'])
    settings_table.add_row(
        "[G] GPU Layers:",
        f"{ngl_display}{ngl_marker}",
        f"[dim]safe: {safe_defaults['gpu_layers']}, total: {usage.get('total_layers', 'N/A')}[/dim]"
    )

    # Flash attention
    fa_status = "✓ ON" if settings["flash_attention"] else "✗ OFF"
    fa_custom = settings["flash_attention"] != (safe_defaults["flash_attention"] != "off")
    fa_marker = " (custom)" if fa_custom else ""
    settings_table.add_row(
        "[F] Flash Attention:",
        f"{fa_status}{fa_marker}",
        f"[dim]reduces memory for large context[/dim]"
    )

    # Cache type
    cache_custom = settings["cache_type"] != safe_defaults["cache_type"]
    cache_marker = " (custom)" if cache_custom else ""
    cache_info = {
        "f16": "no compression (2 bytes/token)",
        "q8_0": "low compression (~1.1 bytes/token)",
        "q4_0": "high compression (~0.6 bytes/token)"
    }
    settings_table.add_row(
        "[K] KV Cache Type:",
        f"{settings['cache_type']}{cache_marker}",
        f"[dim]safe: {safe_defaults['cache_type']}, {cache_info.get(settings['cache_type'], '')}[/dim]"
    )

    # Host, Port, API Key
    settings_table.add_row(
        "[H] Host IP:",
        f"{settings['host']}",
        f"[dim]safe: {safe_defaults.get('host', '0.0.0.0')}[/dim]"
    )
    settings_table.add_row(
        "[P] Port:",
        f"{settings['port']}",
        f"[dim]safe: {safe_defaults.get('port', '8080')}[/dim]"
    )
    settings_table.add_row(
        "[A] API Key:",
        f"{settings.get('api_key') or 'Not set'}",
        f"[dim]safe: {safe_defaults.get('api_key') or 'Not set'}[/dim]"
    )
    
    # Vision model settings (only show if this is a vision model)
    if arch_info.get('has_vision', False):
        settings_table.add_row("", "", "")  # Spacer
        
        # Detect VLM family for contextual hints
        model_name_lower = model_name.lower() if model_name else ""
        arch_name = arch_info.get('architecture', '').lower()
        
        # Tiled high-res readers (resolution CRITICAL)
        is_tiled_reader = any(x in model_name_lower for x in ['minicpm', 'olmocr', 'rolmocr'])
        
        # CLIP-style VLMs (resolution mostly cosmetic)
        is_clip_style = any(x in model_name_lower or x in arch_name for x in ['qwen', 'pixtral', 'llava'])
        
        if is_tiled_reader:
            hint = "[yellow]⚠ CRITICAL for text extraction[/yellow]"
        elif is_clip_style:
            hint = "[dim]mostly cosmetic (auto-resized)[/dim]"
        else:
            hint = "[dim](model load parameters)[/dim]"
        
        settings_table.add_row(
            "[bold cyan]Vision Model Settings:[/bold cyan]",
            "",
            hint
        )
        
        img_res = settings.get('vision_image_resolution', 1024)
        if is_tiled_reader:
            res_hint = f"[yellow]scanner tile size[/yellow]"
        else:
            res_hint = f"[dim]vision encoder tile size[/dim]"
        
        settings_table.add_row(
            "[I] Image Resolution:",
            f"{img_res}",
            res_hint
        )
        
        batch = settings.get('vision_batch_size', 1024)
        settings_table.add_row(
            "[B] Batch Size:",
            f"{batch}",
            f"[dim]memory allocation[/dim]"
        )
        
        ubatch = settings.get('vision_ubatch_size', 512)
        settings_table.add_row(
            "[U] Micro-batch Size:",
            f"{ubatch}",
            f"[dim]processing chunk size[/dim]"
        )
        
        no_offload = settings.get('vision_no_mmproj_offload', False)
        offload_status = "✓ ON" if no_offload else "✗ OFF"
        settings_table.add_row(
            "[O] No Projector Offload:",
            f"{offload_status}",
            f"[dim]keep projector on CPU[/dim]"
        )
    
    # Extra Args (for any other custom flags)
    extra_args_display = settings.get('extra_args', '') or 'Not set'
    if len(extra_args_display) > 50:
        extra_args_display = extra_args_display[:47] + '...'
    settings_table.add_row(
        "[E] Extra Args:",
        f"{extra_args_display}",
        f"[dim]other custom flags[/dim]"
    )

    console.print(settings_table)
    console.print()

    # VRAM breakdown
    vram_table = Table(title="[bold]Estimated VRAM Usage[/bold]", show_header=True, box=None)
    vram_table.add_column("Component", style="bold")
    vram_table.add_column("Size (GB)", justify="right", style="magenta")
    vram_table.add_column("Usage Bar", width=30)

    def make_bar(value: float, max_val: float) -> str:
        bar_width = 30
        if max_val <= 0:
            return "[bright_black]N/A"
        ratio = value / max_val
        filled = int(ratio * bar_width)
        filled = max(0, min(filled, bar_width))
        color = "green" if ratio <= 1.0 else "red"
        return f"[{color}]{'█' * filled}{' ' * (bar_width - filled)}[/{color}]"

    components = {
        "Model Weights": usage.get('model_weights', 0),
        "KV Cache": usage.get('kv_cache', 0),
        "Projector": usage.get('projector_weights', 0),
        "Overhead": usage.get('overhead', 0),
        "Safety Margin (5%)": usage.get('safety_buffer', 0)
    }

    for name, value in components.items():
        if value > 0:
            vram_table.add_row(name, f"{value:.2f}", make_bar(value, vram_available))

    vram_table.add_row(
        "─" * 20,
        "─" * 11,
        "─" * 30
    )

    total = usage['total']
    fits = total <= vram_available
    status_color = "green" if fits else "red"
    status_symbol = "✓" if fits else "✗"
    vram_table.add_row(
        f"[bold {status_color}]Total Estimated[/bold {status_color}]",
        f"[bold {status_color}]{total:.2f} GB[/bold {status_color}]",
        make_bar(total, vram_available)
    )

    if fits:
        vram_table.add_row("[bold green]Remaining[/bold green]", f"[bold green]{vram_free:.2f} GB[/bold green]", "")
    else:
        vram_table.add_row("[bold red]Exceeds by[/bold red]", f"[bold red]{abs(vram_free):.2f} GB[/bold red]", "")

    console.print(vram_table)
    console.print()

    # Status message
    if fits:
        console.print(f"[bold green]{status_symbol} Configuration should fit in VRAM (with safety margin)[/bold green]")
    else:
        console.print(f"[bold red]{status_symbol} WARNING: Configuration likely exceeds available VRAM![/bold red]")
        console.print("[yellow]Reduce context size, GPU layers, or change KV cache quantization.[/yellow]")

    console.print("\n[bold]Options:[/bold] (c)ontext, (g)pu layers, (k)v cache, (r)eset, (l)aunch, (q)uit, (m)anage gpus")

def interactive_vram_config(model_name: str, model_path: str, vram_gb: float,
                           capabilities: Dict[str, Any], safe_defaults: Dict[str, Any],
                           user_defaults: Dict[str, Any], arch_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interactive VRAM configuration loop with accurate calculations.
    Returns final settings dict.
    """
    
    # Initialize selected GPUs (default to [0] or first available)
    available_gpus = capabilities.get("system_flags", {}).get("gpus", [])
    initial_gpus = user_defaults.get("selected_gpus", [0] if not available_gpus else [0])
    
    # Calculate initial VRAM based on standard single GPU if no specific selection logic exists yet,
    # or rely on the passed vram_gb if strictly single-gpu mode was assumed.
    # We will refine this inside the loop.

    settings = {
        "ctx_size": user_defaults.get("ctx_size", safe_defaults.get("ctx_size", 2048)),
        "gpu_layers": user_defaults.get("gpu_layers", safe_defaults.get("gpu_layers", 999)),
        "flash_attention": user_defaults.get("flash_attention", safe_defaults.get("flash_attention", False)),
        "cache_type": user_defaults.get("cache_type", safe_defaults.get("cache_type", "f16")),
        "extra_args": user_defaults.get("extra_args", ""),
        "host": user_defaults.get("host", safe_defaults.get("host", "0.0.0.0")),
        "port": user_defaults.get("port", safe_defaults.get("port", "8080")),
        "api_key": user_defaults.get("api_key", safe_defaults.get("api_key", "")),
        "selected_gpus": initial_gpus,
        # Vision model settings (only used if has_vision=True)
        "vision_image_resolution": user_defaults.get("vision_image_resolution", 1024),
        "vision_batch_size": user_defaults.get("vision_batch_size", 1024),
        "vision_ubatch_size": user_defaults.get("vision_ubatch_size", 512),
        "vision_no_mmproj_offload": user_defaults.get("vision_no_mmproj_offload", False)
    }

    if isinstance(settings["flash_attention"], str):
        settings["flash_attention"] = settings["flash_attention"].lower() == 'on'

    while True:
        # Recalculate total VRAM based on selected GPUs
        current_vram_gb = 0.0
        if available_gpus:
            current_gpus_indices = settings["selected_gpus"]
            for gpu in available_gpus:
                if gpu["index"] in current_gpus_indices:
                    current_vram_gb += gpu["vram_gb"]
            # Fallback if somehow 0 (e.g. no gpus selected), avoids div by zero issues elsewhere if any
            if current_vram_gb == 0 and vram_gb > 0: 
                 # If user deselected everything, functionally they have 0 VRAM for offloading
                 current_vram_gb = 0.0 
        else:
            current_vram_gb = vram_gb

        render_vram_display(model_name, model_path, current_vram_gb, settings, safe_defaults, arch_info, capabilities)

        console.print("[bold]Actions:[/bold]")
        console.print("[C] Adjust Context Size  [G] Adjust GPU Layers  [F] Toggle Flash Attention")
        console.print("[K] Change Cache Type    [R] Reset to Safe Defaults")
        console.print("[H] Set Host IP          [P] Set Port             [A] Set API Key")
        
        # Conditional vision model options
        if arch_info.get('has_vision', False):
            console.print("[I] Image Resolution     [B] Batch Size           [U] Micro-batch Size")
            console.print("[O] Toggle Proj Offload  [E] Extra Args           [M] Manage GPUs")
        else:
            console.print("[E] Extra Args           [M] Manage GPUs")
        
        console.print("[L] Launch Server        [Q] Cancel\n")

        choice = input("Select action: ").strip().lower()

        if choice == 'c':
            max_ctx = arch_info.get('context_length', 131072)
            new_ctx_str = input(f"Context size (current: {settings['ctx_size']}, safe: {safe_defaults['ctx_size']}, max: {max_ctx}): ").strip()
            if new_ctx_str.lower() == 'r':
                settings["ctx_size"] = safe_defaults["ctx_size"]
            elif new_ctx_str.isdigit():
                settings["ctx_size"] = min(int(new_ctx_str), max_ctx)
            else:
                print("Invalid input. Please enter a number.")

        elif choice == 'g':
            total_layers = arch_info.get('n_layers', 40)
            new_ngl = input(f"GPU layers (current: {settings['gpu_layers']}, safe: {safe_defaults['gpu_layers']}, total: {total_layers}): ").strip()
            if new_ngl.lower() == 'all':
                settings['gpu_layers'] = 999
            elif new_ngl.isdigit():
                settings["gpu_layers"] = min(int(new_ngl), total_layers)
            elif new_ngl.lower() == 'r':
                settings["gpu_layers"] = safe_defaults["gpu_layers"]

        elif choice == 'f':
            if capabilities["flash_ok"]:
                settings["flash_attention"] = not settings["flash_attention"]
            else:
                console.print("[yellow]Flash attention not supported by this build.[/yellow]")
                input("Press Enter to continue...")

        elif choice == 'k':
            print("Available cache types: " + ", ".join(_BYTES_PER_ELEM.keys()))
            new_cache = input(f"Cache type (current: {settings['cache_type']}, safe: {safe_defaults['cache_type']}): ").strip().lower()
            if new_cache in _BYTES_PER_ELEM:
                settings["cache_type"] = new_cache
            elif new_cache == 'r':
                settings["cache_type"] = safe_defaults["cache_type"]

        elif choice == 'h':
            new_host = input(f"Host IP (current: {settings['host']}, safe: {safe_defaults.get('host', '0.0.0.0')}): ").strip()
            if new_host.lower() == 'r':
                settings["host"] = safe_defaults.get('host', '0.0.0.0')
            elif new_host:
                settings["host"] = new_host
        
        elif choice == 'p':
            new_port = input(f"Port (current: {settings['port']}, safe: {safe_defaults.get('port', '8080')}): ").strip()
            if new_port.lower() == 'r':
                settings["port"] = safe_defaults.get('port', '8080')
            elif new_port.isdigit():
                settings["port"] = new_port
        
        elif choice == 'a':
            new_api_key = input(f"API Key (current: {'********' if settings.get('api_key') else 'Not set'}): ").strip()
            if new_api_key.lower() == 'r':
                settings["api_key"] = safe_defaults.get('api_key', '')
            elif new_api_key:
                settings["api_key"] = new_api_key
        
        elif choice == 'e':
            console.print("\n[bold cyan]Extra Arguments[/bold cyan]")
            console.print("[dim]Common vision model flags:[/dim]")
            console.print("  --image-resolution 1024")
            console.print("  --batch-size 1024")
            console.print("  --ubatch-size 512")
            console.print("  --no-mmproj-offload")
            console.print("\n[dim]Enter flags separated by spaces (or 'r' to reset, Enter to keep current)[/dim]")
            current_extra = settings.get('extra_args', '')
            console.print(f"Current: {current_extra or 'Not set'}")
            new_extra = input("Extra args: ").strip()
            if new_extra.lower() == 'r':
                settings["extra_args"] = safe_defaults.get('extra_args', '')
                console.print("[green]Reset to safe defaults![/green]")
            elif new_extra:
                settings["extra_args"] = new_extra
                console.print("[green]Extra args updated![/green]")
            input("Press Enter to continue...")
        
        # Vision model settings (conditional)
        elif choice == 'i' and arch_info.get('has_vision', False):
            # Detect VLM family for contextual prompts
            model_name_lower = model_name.lower() if model_name else ""
            is_tiled_reader = any(x in model_name_lower for x in ['minicpm', 'olmocr', 'rolmocr'])
            is_clip_style = any(x in model_name_lower for x in ['qwen', 'pixtral', 'llava'])
            
            console.print("\n[bold]Image Resolution[/bold]")
            if is_tiled_reader:
                console.print("[yellow]⚠ CRITICAL:[/yellow] This model uses tiled high-res reading.")
                console.print("[yellow]Wrong value = missing text or blank output![/yellow]")
                console.print("Recommended: 1024 (document/OCR models)")
            elif is_clip_style:
                console.print("[dim]Note: This model auto-resizes internally.[/dim]")
                console.print("[dim]Changing this value has minimal effect.[/dim]")
                console.print("Typical range: 448-672")
            else:
                console.print("Recommended: 1024")
            
            new_res = input(f"\nEnter resolution (current: {settings['vision_image_resolution']}): ").strip()
            if new_res.isdigit():
                settings["vision_image_resolution"] = int(new_res)
                console.print("[green]Image resolution updated![/green]")
            input("Press Enter to continue...")
        
        elif choice == 'b' and arch_info.get('has_vision', False):
            new_batch = input(f"Batch size (current: {settings['vision_batch_size']}, recommended: 1024): ").strip()
            if new_batch.isdigit():
                settings["vision_batch_size"] = int(new_batch)
                console.print("[green]Batch size updated![/green]")
        
        elif choice == 'u' and arch_info.get('has_vision', False):
            new_ubatch = input(f"Micro-batch size (current: {settings['vision_ubatch_size']}, recommended: 512): ").strip()
            if new_ubatch.isdigit():
                settings["vision_ubatch_size"] = int(new_ubatch)
                console.print("[green]Micro-batch size updated![/green]")
        
        elif choice == 'o' and arch_info.get('has_vision', False):
            settings["vision_no_mmproj_offload"] = not settings.get("vision_no_mmproj_offload", False)
            status = "enabled" if settings["vision_no_mmproj_offload"] else "disabled"
            console.print(f"[green]Projector offload {status}![/green]")
            input("Press Enter to continue...")

        elif choice == 'r':
            settings = {
                "ctx_size": safe_defaults["ctx_size"],
                "gpu_layers": safe_defaults["gpu_layers"],
                "flash_attention": safe_defaults.get("flash_attention", "off") != "off",
                "cache_type": safe_defaults["cache_type"],
                "extra_args": safe_defaults["extra_args"],
                "host": safe_defaults.get('host', '0.0.0.0'),
                "port": safe_defaults.get('port', '8080'),
                "api_key": safe_defaults.get('api_key', ''),
                "selected_gpus": user_defaults.get("selected_gpus", [0] if available_gpus else [])
            }
            console.print("[green]Reset to safe defaults![/green]")
            input("Press Enter to continue...")

        elif choice == 'm':
            if not available_gpus:
                console.print("[yellow]No NVIDIA GPUs detected to manage.[/yellow]")
                input("Press Enter to continue...")
            else:
                while True:
                    console.print("\n[bold cyan]Manage GPUs[/bold cyan]")
                    console.print("Select which GPUs to use (toggle by number):")
                    for gpu in available_gpus:
                        idx = gpu["index"]
                        status = "[green]ON[/green]" if idx in settings["selected_gpus"] else "[dim]OFF[/dim]"
                        console.print(f"[{idx}] {gpu['name']} ({gpu['vram_gb']:.2f} GB) - {status}")
                    
                    console.print("\n(A)ll, (N)one, (D)one")
                    sub_choice = input("Option: ").strip().lower()
                    
                    if sub_choice == 'd':
                        break
                    elif sub_choice == 'a':
                        settings["selected_gpus"] = [g["index"] for g in available_gpus]
                    elif sub_choice == 'n':
                        settings["selected_gpus"] = []
                    elif sub_choice.isdigit():
                        idx = int(sub_choice)
                        if any(g["index"] == idx for g in available_gpus):
                            if idx in settings["selected_gpus"]:
                                settings["selected_gpus"].remove(idx)
                            else:
                                settings["selected_gpus"].append(idx)
                            settings["selected_gpus"].sort()
                        else:
                            console.print("[red]Invalid GPU index[/red]")
                    else:
                        pass
        
        elif choice == 'l':
            usage = calculate_vram_usage(model_path, settings, arch_info)
            # Recalculate available for the check
            current_vram_gb_check = 0.0
            if available_gpus:
                for gpu in available_gpus:
                    if gpu["index"] in settings["selected_gpus"]:
                        current_vram_gb_check += gpu["vram_gb"]
            else:
                current_vram_gb_check = vram_gb
                
            vram_available = get_vram_available(current_vram_gb_check)

            if usage["total"] > vram_available:
                console.print("\n[bold red]WARNING: This configuration may exceed available VRAM![/bold red]")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            return settings

        elif choice == 'q':
            return None