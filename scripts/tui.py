#!/usr/bin/env python3
import os
import sys
import socket
import tempfile
import shutil
from typing import Dict, Any

from .config import load_config, save_config, cache_binary_flags
from .detectors import probe_gpu_support, detect_model_family
from .model_info import render_model_info, read_gguf_metadata
from .server import run_command
from .services.command_builder import ServerCommandBuilder, VisionConfig
from .quantize_merge import list_models, quantize_model, merge_parts, find_split_files
from .utils import normalize_path
from .flags import MODEL_CAPS
from .vram_calculator import interactive_vram_config
from .gguf_parser import get_model_architecture, clear_model_cache, find_projector_file

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Force terminal output to ensure visibility on all systems
console = Console(force_terminal=True)
print("Initializing Llama Manager TUI...", flush=True)


def cleanup_cache_directories():
    """Remove old cache directories created by this tool."""
    temp_dir = tempfile.gettempdir()
    for item in os.listdir(temp_dir):
        if item.startswith("llama_manager_cache_"):
            path = os.path.join(temp_dir, item)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    console.print(f"[dim]Removed old cache directory: {path}[/dim]")
                except Exception as e:
                    console.print(f"[red]Error removing cache directory {path}: {e}[/red]")
    print("DEBUG: cleanup_cache_directories done.", flush=True)


def first_time_setup(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Prompt user for bin and model paths on first run."""
    console.print(Panel("[bold green]--- First Time Setup ---[/bold green]"))
    bin_path = input("Full path to llama.cpp 'build/bin' (where llama-server exists): ").strip().strip('"')
    model_path = input("Full path to folder containing .gguf models: ").strip().strip('"')
    cfg["bin_path"] = normalize_path(bin_path)
    cfg["model_path"] = normalize_path(model_path)
    return cfg


def select_model(cfg: Dict[str, Any]):
    """Display available models and return the selected one."""
    models = list_models(cfg)
    if not models:
        console.print("[bold red]No models found in the model folder![/bold red]")
        return None
    console.print(Panel("[bold cyan]Available Models[/bold cyan]"))
    for i, m in enumerate(models, 1):
        console.print(f"{i}. {m}")
    console.print("0. Back/Cancel")
    selection = input("Select model number (0 to cancel): ")
    if not selection.isdigit():
        console.print("[red]Invalid selection[/red]")
        return None
    sel_num = int(selection)
    if sel_num == 0:
        return None
    if sel_num < 1 or sel_num > len(models):
        console.print("[red]Invalid selection[/red]")
        return None
    return models[sel_num - 1]


def get_model_capabilities(cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Determine what features are actually available for this model on this system.
    Returns a dict with 'flash_ok', 'cache_ok', and other capability flags.
    """
    # Get binary path
    server_path = os.path.join(cfg["bin_path"], "llama-server")
    if os.name == "nt":
        server_path += ".exe"
    
    # Probe GPU and binary capabilities (with caching)
    from .config import CACHE_DIR
    import json
    import socket
    from pathlib import Path
    
    # Check if we have cached flags
    hostname = socket.gethostname()
    cache_file = CACHE_DIR / f"{hostname}_system_flags.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                system_flags = json.load(f)
            
            # Re-probe if cache is stale: no vendor detected, or NVIDIA with no gpus list
            stale = not system_flags.get("vendor") or (
                system_flags.get("vendor") == "NVIDIA" and not system_flags.get("gpus")
            )
            if stale:
                # Stale cache, re-probe
                system_flags = probe_gpu_support(server_path, debug=False)
                with open(cache_file, 'w') as f:
                    json.dump(system_flags, f, indent=4)
        except Exception:
            # If cache is corrupted, re-probe
            system_flags = probe_gpu_support(server_path, debug=False)
            with open(cache_file, 'w') as f:
                json.dump(system_flags, f, indent=4)
    else:
        # First time - probe and cache
        system_flags = probe_gpu_support(server_path, debug=False)
        with open(cache_file, 'w') as f:
            json.dump(system_flags, f, indent=4)
    
    # Determine model family from filename (fast and reliable)
    family = detect_model_family(model_name)
    
    # Get model family capabilities from flags.py
    model_caps = MODEL_CAPS.get(family, MODEL_CAPS["llama"])
    
    # Combine: feature is OK if system supports it AND model family allows it
    capabilities = {
        "family": family,
        "flash_ok": system_flags.get("flash_attn", False) and model_caps.get("flash_attention", False),
        "cache_ok": system_flags.get("cache_ctk", False) and system_flags.get("cache_ctv", False) and model_caps.get("cache_compression", False),
        "ctx_size_ok": system_flags.get("ctx_size_flag", False),
        "gpu_ok": system_flags.get("cuda", False) or system_flags.get("vulkan", False),
        "vram_gb": system_flags.get("vram_gb", 0),
        "vendor": system_flags.get("vendor", "Unknown"),
        "gpu_name": system_flags.get("name", "Unknown"),
        "model_caps": model_caps,
        "system_flags": system_flags
    }
    
    return capabilities


def get_safe_defaults(family: str) -> Dict[str, Any]:
    """Get the safe default values for a model family from flags.py"""
    model_caps = MODEL_CAPS.get(family, MODEL_CAPS["llama"])
    
    return {
        "ctx_size": 8192,
        "gpu_layers": 99,
        "flash_attention": "auto" if model_caps.get("flash_attention", False) else "off",
        "cache_type": "q8_0" if model_caps.get("cache_compression", False) else "f16",
        "cache_enabled": model_caps.get("cache_compression", False),
        "extra_args": ""
    }


def show_model_info(cfg: Dict[str, Any], model_name: str):
    """Render a panel with supported features and recommended settings for the selected model."""
    capabilities = get_model_capabilities(cfg, model_name)
    family = capabilities["family"]
    safe_defaults = get_safe_defaults(family)
    
    # Build the info table
    t = Table.grid(expand=False)
    t.add_column(justify="right", style="bold")
    t.add_column()
    
    t.add_row("Model:", model_name)
    t.add_row("Family:", family)
    t.add_row("GPU:", f"{capabilities['vendor']} {capabilities['gpu_name']} ({capabilities['vram_gb']} GB VRAM)")
    t.add_row("", "")
    t.add_row("Flash Attention (binary+model+gpu):", "✅" if capabilities["flash_ok"] else "❌")
    t.add_row("Cache Compression:", "✅" if capabilities["cache_ok"] else "❌")
    t.add_row("Context Size Support:", "✅" if capabilities["ctx_size_ok"] else "❌")
    t.add_row("GPU Offload:", "✅" if capabilities["gpu_ok"] else "❌")
    t.add_row("", "")
    t.add_row("Safe Defaults:", "")
    t.add_row("  Context Size:", str(safe_defaults["ctx_size"]))
    t.add_row("  GPU Layers:", str(safe_defaults["gpu_layers"]))
    t.add_row("  Flash Attention:", safe_defaults["flash_attention"])
    t.add_row("  Cache Type:", safe_defaults["cache_type"])
    
    note = capabilities["model_caps"].get("note", "")
    if note:
        t.add_row("", "")
        t.add_row("Note:", note)
    
    panel = Panel(t, title="Model Info & Capabilities")
    console.print(panel)


def prompt_for_setting(setting_name: str, current_value: Any, safe_default: Any, 
                       options: list = None, is_custom: bool = False) -> tuple:
    """
    Prompt user for a setting value. (Deprecated - kept for compatibility)
    Returns (new_value, was_reset)
    """
    # Build the prompt
    if is_custom and current_value != safe_default:
        prompt_text = f"{setting_name}: {current_value} (custom) [safe default: {safe_default}]\n"
        prompt_text += "Type new value, 'r' to reset to safe, or Enter to keep custom: "
    else:
        prompt_text = f"{setting_name} ({safe_default}): "
        if options:
            prompt_text = f"{setting_name} ({safe_default}) - options: {', '.join(map(str, options))}\n"
            prompt_text += "Type new value or Enter to keep: "
        else:
            prompt_text += "Type new value or Enter to keep: "
    
    user_input = input(prompt_text).strip()
    
    # Handle reset
    if user_input.lower() == 'r':
        console.print(f"[green]Reset to safe default: {safe_default}[/green]")
        return safe_default, True
    
    # Handle empty (keep current)
    if not user_input:
        return current_value if is_custom else safe_default, False
    
    # Validate options if provided
    if options and user_input not in options:
        console.print(f"[yellow]Invalid option. Using safe default: {safe_default}[/yellow]")
        return safe_default, True
    
    # Return new value
    return user_input, False


def interactive_launch(cfg: Dict[str, Any], model_name: str):
    """
    Interactive VRAM-aware launch system with accurate GGUF parsing.
    Shows live VRAM calculations as user adjusts settings.
    """
    capabilities = get_model_capabilities(cfg, model_name)
    family = capabilities["family"]
    safe_defaults = get_safe_defaults(family)
    
    # Get current user defaults from config (model-specific)
    user_defaults = cfg.get("model_defaults", {}).get(model_name, {})
    
    # Get VRAM amount
    vram_gb = capabilities.get("vram_gb", 12)  # Default to 12GB if unknown
    
    # Parse GGUF file for accurate architecture info
    model_path = os.path.join(cfg["model_path"], model_name)
    arch_info = get_model_architecture(model_path)

    # Check for vision capabilities and find projector file
    projector_file = None
    if arch_info.get("has_vision", False):
        projector_file = find_projector_file(model_path)
        if projector_file:
            console.print(f"[cyan]Vision model detected - found projector file:[/cyan]")
            console.print(f"[dim]{projector_file}[/dim]")
        else:
            console.print("[yellow]Vision model detected but no projector file found.[/yellow]")
            console.print("[yellow]Vision features may not work without a .mmproj file.[/yellow]")

    # Launch interactive calculator
    settings = interactive_vram_config(
        model_name,
        model_path,
        vram_gb,
        capabilities,
        safe_defaults,
        user_defaults,
        arch_info
    )

    # If user cancelled
    if settings is None:
        console.print("[yellow]Launch cancelled[/yellow]")
        return

    # Ask to save as new defaults
    console.print("\n[bold]Save these settings as new defaults for this model?[/bold]")
    save_choice = input("(y/n, default: n): ").strip().lower()

    if save_choice == 'y':
        # Save to model_defaults
        if "model_defaults" not in cfg:
            cfg["model_defaults"] = {}
        cfg["model_defaults"][model_name] = {
            "ctx_size": settings["ctx_size"],
            "gpu_layers": settings["gpu_layers"],
            "flash_attention": "on" if settings["flash_attention"] else "off",
            "cache_type": settings["cache_type"],
            "extra_args": settings.get("extra_args", ""),
            "host": settings.get("host"),
            "port": settings.get("port"),
            "api_key": settings.get("api_key"),
            "selected_gpus": settings.get("selected_gpus"),
            # Vision model settings
            "vision_image_resolution": settings.get("vision_image_resolution", 1024),
            "vision_batch_size": settings.get("vision_batch_size", 1024),
            "vision_ubatch_size": settings.get("vision_ubatch_size", 512),
            "vision_no_mmproj_offload": settings.get("vision_no_mmproj_offload", False)
        }
        save_config(cfg)
        console.print("[green]Settings saved as new defaults for this model![/green]")

    # Build and run the command
    console.print("\n[bold green]Starting server...[/bold green]")

    # Create a temporary directory for the cache
    cache_dir = tempfile.mkdtemp(prefix="llama_manager_cache_")
    console.print(f"[dim]Using cache directory: {cache_dir}[/dim]")

    # Prepare vision settings if this is a vision model
    vision_config = None
    if arch_info.get("has_vision", False):
        vision_config = VisionConfig(
            image_resolution=settings.get("vision_image_resolution", 1024),
            batch_size=settings.get("vision_batch_size", 1024),
            ubatch_size=settings.get("vision_ubatch_size", 512),
            no_mmproj_offload=settings.get("vision_no_mmproj_offload", False)
        )
    
    builder = ServerCommandBuilder(cfg)
    cmd = builder.build(
        model_file=model_name,
        ctx_size=settings["ctx_size"],
        gpu_layers=str(settings["gpu_layers"]),
        flash=settings["flash_attention"],
        cache=(settings["cache_type"] != "f16"),
        cache_type=settings["cache_type"],
        extra_args=settings.get("extra_args", ""),
        ngl_override=str(settings["gpu_layers"]),
        cache_dir=cache_dir,
        host=settings.get("host"),
        port=settings.get("port"),
        api_key=settings.get("api_key"),
        projector_file=projector_file,
        vision_config=vision_config
    )

    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")
    
    # Prepare environment with selected GPUs
    env = {}
    if "selected_gpus" in settings and settings["selected_gpus"]:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, settings["selected_gpus"]))
        console.print(f"[dim]Environment: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}[/dim]")
        
    run_command(cmd, env=env)


def reset_model_defaults(cfg: Dict[str, Any], model_name: str = None):
    """Reset settings to safe defaults for a specific model or all models."""
    if model_name:
        if "model_defaults" in cfg and model_name in cfg["model_defaults"]:
            del cfg["model_defaults"][model_name]
            save_config(cfg)
            console.print(f"[green]Reset settings for {model_name} to safe defaults![/green]")
        else:
            console.print(f"[yellow]No custom settings found for {model_name}[/yellow]")
    else:
        if "model_defaults" in cfg:
            cfg["model_defaults"] = {}
            save_config(cfg)
            console.print("[green]Reset ALL model settings to safe defaults![/green]")
        else:
            console.print("[yellow]No custom settings to reset[/yellow]")


def main():
    print("DEBUG: Calling cleanup_cache_directories...", flush=True)
    cleanup_cache_directories()
    print("DEBUG: Loading config...", flush=True)
    cfg = load_config()
    print(f"DEBUG: Config loaded. bin={bool(cfg.get('bin_path'))}, model={bool(cfg.get('model_path'))}", flush=True)

    # First-time setup if needed
    if not cfg.get("bin_path") or not cfg.get("model_path"):
        print("DEBUG: Starting first_time_setup...", flush=True)
        cfg = first_time_setup(cfg)
        save_config(cfg)
    
    print("DEBUG: Entering main loop...", flush=True)
    while True:
        try:
            console.print(Panel("[bold cyan]=== Llama Manager CLI ===[/bold cyan]"))
        except Exception as e:
            print(f"DEBUG: Error printing panel: {e}", flush=True)
            
        console.print("1. Start Server")
        console.print("2. Quantize a Model")
        console.print("3. Merge Split Models")
        console.print("4. Show Model Info")
        console.print("5. Reset Model Settings to Safe Defaults")
        console.print("6. Clear Model Architecture Cache")
        console.print("7. Edit Paths / Reset Config")
        console.print("8. Clear System/GPU Cache")
        console.print("9. Exit")

        print("DEBUG: Waiting for user input...", flush=True)
        choice = input("\nSelect option: ").strip()
        print(f"DEBUG: Input received: {choice}", flush=True)
        
        if choice == "1":
            model_name = select_model(cfg)
            if model_name:
                interactive_launch(cfg, model_name)
                
        elif choice == "2":
            model_name = select_model(cfg)
            if model_name:
                console.print("\nAvailable quantization types:")
                console.print("q4_0, q4_1, q5_0, q5_1, q8_0, q4_k_m, q5_k_m, q6_k")
                q_type = input("Enter quantization type (or Enter to cancel): ").strip()
                if not q_type:
                    console.print("[yellow]Quantization cancelled[/yellow]")
                    continue
                out_name = input("Output filename (leave empty for auto): ").strip() or None
                try:
                    quantize_model(cfg, model_name, q_type, out_name)
                    console.print("[green]Quantization complete![/green]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    
        elif choice == "3":
            split_files = find_split_files(cfg)
            if not split_files:
                console.print("[yellow]No split model files found[/yellow]")
            else:
                console.print("\nFound split models:")
                for i, f in enumerate(split_files, 1):
                    console.print(f"{i}. {os.path.basename(f)}")
                console.print("0. Back/Cancel")
                sel = input("Select file number (0 to cancel): ").strip()
                if sel.isdigit():
                    sel_num = int(sel)
                    if sel_num == 0:
                        continue
                    if 1 <= sel_num <= len(split_files):
                        first_part = split_files[sel_num - 1]
                        out_name = input("Output filename (or Enter to cancel): ").strip()
                        if not out_name:
                            console.print("[yellow]Merge cancelled[/yellow]")
                            continue
                        try:
                            merge_parts(cfg, first_part, out_name)
                            console.print("[green]Merge complete![/green]")
                        except Exception as e:
                            console.print(f"[red]Error: {e}[/red]")
                        
        elif choice == "4":
            model_name = select_model(cfg)
            if model_name:
                show_model_info(cfg, model_name)
                input("\nPress Enter to continue...")
                
        elif choice == "5":
            console.print("\n1. Reset specific model")
            console.print("2. Reset ALL models")
            console.print("0. Back/Cancel")
            reset_choice = input("Select option: ").strip()
            if reset_choice == "1":
                model_name = select_model(cfg)
                if model_name:
                    reset_model_defaults(cfg, model_name)
            elif reset_choice == "2":
                confirm = input("Reset ALL model settings? (yes/no): ").strip().lower()
                if confirm == "yes":
                    reset_model_defaults(cfg)
            elif reset_choice == "0":
                continue
            input("\nPress Enter to continue...")
            
        elif choice == "6":
            console.print("\n1. Clear specific model cache")
            console.print("2. Clear ALL model caches")
            console.print("0. Back/Cancel")
            cache_choice = input("Select option: ").strip()
            if cache_choice == "1":
                model_name = select_model(cfg)
                if model_name:
                    clear_model_cache(model_name)
                    console.print(f"[green]Cleared cache for {model_name}[/green]")
            elif cache_choice == "2":
                confirm = input("Clear ALL model caches? (yes/no): ").strip().lower()
                if confirm == "yes":
                    clear_model_cache()
                    console.print("[green]Cleared all model caches![/green]")
            elif cache_choice == "0":
                continue
            input("\nPress Enter to continue...")
            
        elif choice == "7":
            cfg = first_time_setup(cfg)
            save_config(cfg)
            
        elif choice == "8":
            from .config import CACHE_DIR
            hostname = socket.gethostname()
            cache_file = CACHE_DIR / f"{hostname}_system_flags.json"
            if cache_file.exists():
                cache_file.unlink()
                console.print("[green]System/GPU cache cleared. Please restart the application.[/green]")
            else:
                console.print("[yellow]No system/GPU cache to clear.[/yellow]")
            input("\nPress Enter to continue...")

        elif choice == "9":
            console.print("[bold green]Goodbye![/bold green]")
            sys.exit(0)