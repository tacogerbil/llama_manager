import os
import shlex
from typing import List




def normalize_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p.strip().strip('"'))) if p else p




def safe_split_extra(extra: str) -> List[str]:
    try:
        return shlex.split(extra)
    except Exception:
        return extra.split()