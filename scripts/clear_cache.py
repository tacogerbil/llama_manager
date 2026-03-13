
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
from scripts.gguf_parser import clear_model_cache

if __name__ == "__main__":
    print("Clearing model cache...")
    clear_model_cache()
    print("Done.")
