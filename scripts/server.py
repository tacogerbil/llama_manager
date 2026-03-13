import os
import shlex
import subprocess
import platform
from typing import List, Dict, Any
from typing import List, Dict, Any






def run_command(cmd: List[str], background: bool = True, env: Dict[str, str] = None):
    """
    Run command with proper Windows/Linux handling.
    Fixed to work correctly in PowerShell and CMD.
    """
    system = platform.system()
    
    # Merge with current environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    if background:
        if system == "Windows":
            # Use CREATE_NEW_CONSOLE to open in new window
            # This works in both PowerShell and CMD
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                # Don't use shell=True - it causes issues with paths containing spaces
                shell=False,
                env=full_env
            )
        else:
            # Linux/Unix - start in new session
            subprocess.Popen(cmd, start_new_session=True, env=full_env)
    else:
        # Foreground execution
        subprocess.run(cmd, check=True, env=full_env)