from scripts.detectors import probe_gpu_support
import json

print("Probing GPUs...")
try:
    flags = probe_gpu_support(debug=True)
    print("\n--- Result ---")
    print(json.dumps(flags, indent=2, default=str))
    
    gpus = flags.get("system_flags", {}).get("gpus", []) if "system_flags" in flags else flags.get("gpus", [])
    # Note: probe_gpu_support returns the flags dict directly, which matches the 'system_flags' structure in config but here it IS the dict.
    
    print(f"\nDetected {len(flags.get('gpus', []))} GPUs in 'gpus' list.")
    
except Exception as e:
    print(f"Error: {e}")
