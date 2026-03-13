# Central place for model family defaults and capability table


MODEL_CAPS = {
    "gemma": {
        "flash_attention": False,
        "cache_compression": False,
        "gpu_offload": "safe",
        "note": "Gemma: disable flash & cache by default"
        },
    "mixtral": {
        "flash_attention": False,
        "cache_compression": True,
        "gpu_offload": "safe",
        "note": "Mixtral: FA sometimes unsupported"
        },
    "qwen": {
        "flash_attention": True,
        "cache_compression": True,
        "gpu_offload": "max",
        "note": "Qwen: supports FA & cache"
        },
    "mistral": {
        "flash_attention": True,
        "cache_compression": True,
        "gpu_offload": "max",
        "note": "Mistral"
        },
    "llama": {
        "flash_attention": True,
        "cache_compression": True,
        "gpu_offload": "max",
        "note": "LLaMA family"
        },
    "llava": {
        "flash_attention": True,
        "cache_compression": True,
        "gpu_offload": "max",
        "note": "LLaVA vision model - requires .mmproj file"
        },
    "minicpm": {
        "flash_attention": True,
        "cache_compression": True,
        "gpu_offload": "max",
        "note": "MiniCPM-V vision model - requires .mmproj file"
        }
}