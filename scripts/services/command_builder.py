import os
import shlex
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scripts.detectors import detect_model_family

@dataclass
class VisionConfig:
    """Type-safe configuration for Vision Models."""
    image_resolution: int = 1024
    batch_size: int = 1024
    ubatch_size: int = 512
    no_mmproj_offload: bool = False
    
    # MiniCPM Specifics
    disable_thinking: bool = False
    force_jinja: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisionConfig':
        """Factory specific to handling keys from VisionSettingsWidget."""
        return cls(
            image_resolution=int(data.get("image_resolution", 1024)),
            batch_size=int(data.get("batch_size", 1024)),
            ubatch_size=int(data.get("ubatch_size", 512)),
            no_mmproj_offload=data.get("no_mmproj_offload", False),
            disable_thinking=data.get("disable_thinking", False),
            force_jinja=data.get("force_jinja", False)
        )

class ServerCommandBuilder:
    """
    Builder for llama-server subprocess commands.
    MCCC: Encapsulates command-line flag logic, separated from GUI.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def build(self, 
              model_file: str, 
              ctx_size: int, 
              gpu_layers: str,
              flash: bool, 
              cache: bool, 
              cache_type: str, 
              extra_args: str,
              ngl_override: Optional[str] = None, 
              cache_dir: Optional[str] = None,
              host: Optional[str] = None, 
              port: Optional[str] = None, 
              api_key: Optional[str] = None, 
              projector_file: Optional[str] = None, 
              vision_config: Optional[VisionConfig] = None) -> List[str]:
        
        # 1. Base Command
        server_path = os.path.join(self.cfg["bin_path"], "llama-server")
        if os.name == "nt": 
            server_path += ".exe"
            
        model_path = os.path.join(self.cfg["model_path"], model_file)
        cmd = [server_path, "-m", model_path]

        # 2. Network / Auth
        if host: cmd.extend(["--host", host])
        if port: cmd.extend(["--port", str(port)])
        if api_key: cmd.extend(["--api-key", api_key])

        # 3. Model-family chat template (unconditional — must not depend on vision pipeline)
        family = detect_model_family(model_file)
        if family == "minicpm":
            cmd.extend(["--chat-template", "chatml"])
            cmd.extend(["-r", "<|im_end|>,<|endoftext|>"])

        # 4. Vision / Multimodal
        if projector_file:
            cmd.extend(["--mmproj", projector_file])
            if vision_config:
                self._apply_vision_args(cmd, vision_config, model_file)

        # 6. Compute / Hardware
        if ngl_override:
            cmd.extend(["-ngl", str(ngl_override)])
        else:
            default_ngl = self.cfg.get("defaults", {}).get("gpu_layers", "99")
            cmd.extend(["-ngl", str(default_ngl)])

        if ctx_size:
            cmd.extend(["--ctx-size", str(ctx_size)])

        if flash:
            cmd.extend(["-fa", "on"])

        if cache and cache_type:
            cmd.extend(["-ctk", cache_type, "-ctv", cache_type])
            
        if cache_dir:
            cmd.extend(["--slot-save-path", cache_dir])

        # 5. Extra User Args
        if extra_args:
            try:
                extra_list = shlex.split(extra_args)
                cmd.extend(extra_list)
            except:
                cmd.extend(extra_args.split())

        return cmd

    def _apply_vision_args(self, cmd: List[str], config: VisionConfig, model_file: str):
        """Applies vision-specific flags, including model-specific strategies."""
        
        # Standard Vision Flags
        if config.batch_size:
            cmd.extend(["--batch-size", str(config.batch_size)])
            
        if config.ubatch_size:
            cmd.extend(["--ubatch-size", str(config.ubatch_size)])
            
        if config.no_mmproj_offload:
            cmd.append("--no-mmproj-offload")

        # Model-Specific Strategy (MiniCPM)
        family = detect_model_family(model_file)
        if family == "minicpm":
             self._apply_minicpm_strategy(cmd, config)

    def _apply_minicpm_strategy(self, cmd: List[str], config: VisionConfig):
        """Encapsulated MiniCPM specific logic."""

        # Optional: Force Jinja engine (uses GGUF template via jinja parser, no template erase)
        if config.force_jinja:
            cmd.extend([
                "--jinja",
                "--reasoning-format", "none",
            ])

        # Disable Thinking
        if config.disable_thinking:
            cmd.extend(["--reasoning-budget", "0"])
