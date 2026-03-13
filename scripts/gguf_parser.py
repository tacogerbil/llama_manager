import struct
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, List


class GGUFParser:
    """Parse GGUF file metadata to extract model architecture information."""
    
    # GGUF value types
    GGUF_VALUE_TYPE_UINT8 = 0
    GGUF_VALUE_TYPE_INT8 = 1
    GGUF_VALUE_TYPE_UINT16 = 2
    GGUF_VALUE_TYPE_INT16 = 3
    GGUF_VALUE_TYPE_UINT32 = 4
    GGUF_VALUE_TYPE_INT32 = 5
    GGUF_VALUE_TYPE_FLOAT32 = 6
    GGUF_VALUE_TYPE_BOOL = 7
    GGUF_VALUE_TYPE_STRING = 8
    GGUF_VALUE_TYPE_ARRAY = 9
    GGUF_VALUE_TYPE_UINT64 = 10
    GGUF_VALUE_TYPE_INT64 = 11
    GGUF_VALUE_TYPE_FLOAT64 = 12
    
    MAX_STR_LEN = 10 * 1024 * 1024  # 10 MB limit for strings
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.metadata = {}
        
    def parse(self) -> Dict[str, Any]:
        """Parse the GGUF file and return metadata dict."""
        if not self.filepath.exists():
            return {}
        
        # print(f"DEBUG: Parsing {self.filepath.name}")

        try:
            with open(self.filepath, 'rb') as f:
                # Read magic
                magic = f.read(4)
                if magic != b'GGUF':
                    print(f"DEBUG: Invalid magic: {magic} in {self.filepath.name}")
                    return {}
                
                # Read version
                version = struct.unpack('<I', f.read(4))[0]
                if version not in [2, 3]:  # Support GGUF v2 and v3
                    print(f"DEBUG: Unsupported version {version} in {self.filepath.name}")
                    return {}
                
                # Check file size to avoid reading past end
                file_size = self.filepath.stat().st_size
                
                # Read tensor and metadata counts
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                
                # print(f"DEBUG: Ver {version}, Tensors {tensor_count}, Metadata {metadata_count}")
                
                # Sanity check counts
                if metadata_count > 100000: # heuristic limit
                     print(f"Warning: Abnormal metadata count {metadata_count} in {self.filepath.name}")
                     return {}

                # Parse metadata key-value pairs
                for i in range(metadata_count):
                    try:
                        file_pos = f.tell()
                        key = self._read_string(f, file_pos)
                        value_type = struct.unpack('<I', f.read(4))[0]
                        value = self._read_value(f, value_type, file_pos)
                        
                        if key and value is not None:
                            self.metadata[key] = value
                    except ValueError as ve:
                        # Log the specific metadata entry that failed and skip it
                        print(f"Warning: Skipping corrupted metadata entry {i+1}/{metadata_count} at position {file_pos} in {self.filepath.name}: {ve}")
                        # Try to recover by returning what we have so far
                        if self.metadata:
                            return self._extract_model_info()
                        return {}
                
                return self._extract_model_info()
                
        except MemoryError:
            print(f"Error parsing GGUF: MemoryError (Possible corrupted file or invalid format) - {self.filepath.name}")
            return {}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error parsing GGUF: {e} in file {self.filepath.name}")
            return {}
    
    def _read_string(self, f, file_pos: int = None) -> str:
        """Read a GGUF string (length-prefixed)."""
        length = struct.unpack('<Q', f.read(8))[0]
        if length > self.MAX_STR_LEN:
            pos_info = f" at position {file_pos}" if file_pos is not None else ""
            raise ValueError(f"String length {length} exceeds max limit {self.MAX_STR_LEN}{pos_info} in {self.filepath.name}")
        return f.read(length).decode('utf-8', errors='ignore')
    
    def _read_value(self, f, value_type: int, file_pos: int = None):
        """Read a value based on its type."""
        if value_type == self.GGUF_VALUE_TYPE_UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == self.GGUF_VALUE_TYPE_INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == self.GGUF_VALUE_TYPE_UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == self.GGUF_VALUE_TYPE_INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == self.GGUF_VALUE_TYPE_UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == self.GGUF_VALUE_TYPE_INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == self.GGUF_VALUE_TYPE_FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == self.GGUF_VALUE_TYPE_UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == self.GGUF_VALUE_TYPE_INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == self.GGUF_VALUE_TYPE_FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif value_type == self.GGUF_VALUE_TYPE_BOOL:
            return struct.unpack('<B', f.read(1))[0] != 0
        elif value_type == self.GGUF_VALUE_TYPE_STRING:
            return self._read_string(f, file_pos)
        elif value_type == self.GGUF_VALUE_TYPE_ARRAY:
            # Skip arrays for now (complex to parse)
            array_type = struct.unpack('<I', f.read(4))[0]
            array_len = struct.unpack('<Q', f.read(8))[0]
            
            # Sanity check array length
            if array_len > 1000000:  # 1M elements max
                pos_info = f" at position {file_pos}" if file_pos is not None else ""
                raise ValueError(f"Array length {array_len} exceeds reasonable limit{pos_info}")
            
            # Skip array data
            element_size = self._get_type_size(array_type)
            if element_size > 0:
                f.read(element_size * array_len)
            elif array_type == self.GGUF_VALUE_TYPE_STRING:
                # String arrays need special handling - skip each string individually
                for _ in range(array_len):
                    str_len = struct.unpack('<Q', f.read(8))[0]
                    if str_len > self.MAX_STR_LEN:
                        raise ValueError(f"String in array exceeds max limit")
                    f.read(str_len)  # Skip the string data
            return None
        else:
            return None
    
    def _get_type_size(self, value_type: int) -> int:
        """Get size in bytes for a value type."""
        sizes = {
            0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 
            6: 4, 7: 1, 10: 8, 11: 8, 12: 8
        }
        return sizes.get(value_type, 0)
    
    def _extract_model_info(self) -> Dict[str, Any]:
        """Extract relevant model architecture info from metadata."""
        info = {
            'architecture': None,
            'hidden_size': None,
            'n_heads': None,
            'n_kv_heads': None,
            'n_layers': None,
            'context_length': None,
            'vocab_size': None,
            'file_size_gb': None,
            'has_vision': False
        }

        # Try to find architecture
        for key in self.metadata:
            if 'general.architecture' in key:
                info['architecture'] = self.metadata[key]
                break

        # Detect vision capabilities from GGUF metadata
        vision_keys = [
            'clip.has_vision_encoder',
            'clip.has_text_encoder',
            'vision.image_size',
            'vision.patch_size',
            'vision.projection_dim',
            'llava.projector_type',
            'mm.projector_type'
        ]

        for key in self.metadata:
            if any(vkey in key for vkey in vision_keys):
                # Check if it's a boolean and true, or just presence of vision keys
                value = self.metadata[key]
                if isinstance(value, bool):
                    if value:
                        info['has_vision'] = True
                        break
                else:
                    # Non-boolean vision keys indicate vision support
                    info['has_vision'] = True
                    break
        
        # Explicit architecture checks for Vision models
        # Qwen2-VL often uses 'qwen2vl' architecture but might lack generic vision keys
        if info['architecture'] == 'qwen2vl':
            info['has_vision'] = True

        # Robust Fallback: Check filename for "vl" or "minicpm-v" patterns
        # "vl" catches Qwen-VL etc; "minicpm-v" catches MiniCPM-V which uses "-V-" not "vl"
        name_lower = self.filepath.name.lower()
        if not info['has_vision'] and ("vl" in name_lower or "minicpm-v" in name_lower):
            info['has_vision'] = True
        
        # Find the architecture prefix (e.g., "llama", "qwen2")
        arch_prefix = info['architecture']
        if not arch_prefix:
            return info
        
        # Map common metadata keys
        key_mappings = {
            'hidden_size': [f'{arch_prefix}.embedding_length', f'{arch_prefix}.hidden_size'],
            'n_heads': [f'{arch_prefix}.attention.head_count', f'{arch_prefix}.n_head'],
            'n_kv_heads': [f'{arch_prefix}.attention.head_count_kv', f'{arch_prefix}.n_head_kv'],
            'n_layers': [f'{arch_prefix}.block_count', f'{arch_prefix}.n_layer'],
            'context_length': [f'{arch_prefix}.context_length', f'{arch_prefix}.n_ctx'],
            'vocab_size': [f'{arch_prefix}.vocab_size']
        }
        
        # Extract values
        for info_key, possible_keys in key_mappings.items():
            for metadata_key in possible_keys:
                if metadata_key in self.metadata:
                    info[info_key] = self.metadata[metadata_key]
                    break
        
        # Get file size
        try:
            info['file_size_gb'] = self.filepath.stat().st_size / (1024**3)
        except:
            pass
        
        # If n_kv_heads not found, assume it equals n_heads (standard attention)
        if info['n_kv_heads'] is None and info['n_heads'] is not None:
            info['n_kv_heads'] = info['n_heads']
        
        return info


def get_model_architecture(model_path: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Parse GGUF file and return architecture information.
    Caches results based on file modification time.
    Falls back to estimates if parsing fails.
    
    Args:
        model_path: Path to GGUF model file
        use_cache: If True, use cached results if available and file unchanged
    """
    model_file = Path(model_path)
    
    # Check cache first
    if use_cache:
        cache_dir = Path(__file__).parent.parent / "config" / "model_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache filename based on model filename
        cache_file = cache_dir / f"{model_file.name}.json"
        
        if cache_file.exists() and model_file.exists():
            try:
                # Check if model file has been modified since cache was created
                model_mtime = model_file.stat().st_mtime
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verify cache is still valid
                if cached_data.get('_cache_mtime') == model_mtime:
                    # Remove internal cache metadata before returning
                    cached_data.pop('_cache_mtime', None)
                    return cached_data
            except Exception:
                pass  # Cache read failed, will re-parse
    
    # Parse GGUF file
    parser = GGUFParser(model_path)
    info = parser.parse()
    
    # If parsing succeeded and we have key info, cache it
    if info.get('n_layers') and info.get('hidden_size'):
        # DOUBLE CHECK: If filename says VL or minicpm-v, force has_vision = True
        # This caches the correct value so we don't need to re-parse
        _name_lower = model_file.name.lower()
        if not info.get('has_vision') and ("vl" in _name_lower or "minicpm-v" in _name_lower):
            info['has_vision'] = True

        if use_cache and model_file.exists():
            try:
                cache_dir = Path(__file__).parent.parent / "config" / "model_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = cache_dir / f"{model_file.name}.json"
                
                # Add modification time to cache
                info['_cache_mtime'] = model_file.stat().st_mtime
                
                with open(cache_file, 'w') as f:
                    json.dump(info, f, indent=2)
                
                # Remove internal metadata before returning
                info.pop('_cache_mtime', None)
            except Exception:
                pass  # Cache write failed, continue without caching
        
        return info
    
    # Fallback to filename-based estimation
    model_name = model_file.name.lower()
    
    # Default fallback values
    fallback = {
        'architecture': 'unknown',
        'hidden_size': 4096,
        'n_heads': 32,
        'n_kv_heads': 32,
        'n_layers': 32,
        'context_length': 8192,
        'vocab_size': 32000,
        'file_size_gb': None
    }
    
    # Try to get file size at least
    try:
        fallback['file_size_gb'] = model_file.stat().st_size / (1024**3)
    except:
        pass
    
    # Estimate from filename
    if "qwen" in model_name:
        fallback['architecture'] = 'qwen2'
        if "14b" in model_name:
            fallback.update({
                'hidden_size': 5120,
                'n_heads': 40,
                'n_kv_heads': 8,  # Qwen2.5 uses GQA
                'n_layers': 40,
                'context_length': 32768
            })
            fallback.update({
                'hidden_size': 4096,
                'n_heads': 32,
                'n_kv_heads': 8,
                'n_layers': 32,
                'context_length': 32768
            })
        
        if "vl" in model_name:
             fallback['has_vision'] = True
    elif "llama" in model_name:
        fallback['architecture'] = 'llama'
        if "70b" in model_name or "72b" in model_name:
            fallback.update({
                'hidden_size': 8192,
                'n_heads': 64,
                'n_kv_heads': 8,
                'n_layers': 80
            })
        elif "13b" in model_name:
            fallback.update({
                'hidden_size': 5120,
                'n_heads': 40,
                'n_kv_heads': 40,
                'n_layers': 40
            })
    elif "gemma" in model_name:
        fallback['architecture'] = 'gemma'
        if "7b" in model_name:
            fallback.update({
                'hidden_size': 3072,
                'n_heads': 16,
                'n_kv_heads': 16,
                'n_layers': 28
            })
    
    # Cache fallback results too
    if use_cache and model_file.exists():
        try:
            cache_dir = Path(__file__).parent.parent / "config" / "model_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{model_file.name}.json"
            
            fallback['_cache_mtime'] = model_file.stat().st_mtime
            fallback['_is_fallback'] = True
            
            with open(cache_file, 'w') as f:
                json.dump(fallback, f, indent=2)
            
            fallback.pop('_cache_mtime', None)
            fallback.pop('_is_fallback', None)
        except Exception:
            pass
    
    return fallback


def list_projector_candidates(model_path: str) -> List[str]:
    """
    Return a list of all potential projector files in the model's directory.
    Useful for populating a dropdown.
    """
    model_file = Path(model_path)
    model_dir = model_file.parent
    
    # search for anything looking like a projector
    candidates = list(model_dir.glob("mmproj-*.gguf")) + \
                 list(model_dir.glob("*-mmproj.gguf")) + \
                 list(model_dir.glob("*.mmproj"))
                 
    # clean and sort
    candidates = sorted(list(set([str(c) for c in candidates])))
    return candidates

def find_projector_file(model_path: str) -> str | None:
    """
    Find the best matching projector file for a vision model.
    """
    # Use the candidates list logic
    candidates = list_projector_candidates(model_path)
    if not candidates:
        return None
        
    model_file = Path(model_path)
    model_name = model_file.stem
    
    # Clean model name for matching
    clean_model_name = re.sub(r'[-_](?:[qQ]\d+(?:_[0-9A-Za-z]+)?|f(?:16|32)|bf16|iq\d+.*?)$', '', model_name, flags=re.IGNORECASE)
    
    # Filter for best match
    matches = []
    for cand in candidates:
        cand_name = Path(cand).name
        if clean_model_name.lower() in cand_name.lower():
            matches.append(cand)
            
    if matches:
        return matches[0]
        
    return candidates[0] if candidates else None


def clear_model_cache(model_name: str = None):
    """
    Clear cached model architecture data.

    Args:
        model_name: If provided, clear cache for specific model.
                   If None, clear all cached models.
    """
    cache_dir = Path(__file__).parent.parent / "config" / "model_cache"

    if not cache_dir.exists():
        return

    try:
        if model_name:
            # Clear specific model
            cache_file = cache_dir / f"{model_name}.json"
            if cache_file.exists():
                cache_file.unlink()
        else:
            # Clear all cached models
            for cache_file in cache_dir.glob("*.json"):
                cache_file.unlink()
            # Optionally remove the directory if empty
            if not any(cache_dir.iterdir()):
                cache_dir.rmdir()
    except Exception as e:
        print(f"Error clearing cache: {e}")