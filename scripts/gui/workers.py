from PySide6.QtCore import QObject, QProcess, Signal, QTimer
import os
import re

class ServerSession(QObject):
    """
    Manages a single llama-server instance.
    Wraps QProcess to handle specific logs and lifecycle events.
    """
    # Signals
    log_received = Signal(str, str) # (port, message)
    status_changed = Signal(str, bool) # (port, is_running)
    finished = Signal(str, int) # (port, exit_code)
    stats_updated = Signal(str, dict) # (port, stats)

    def __init__(self, port, model_name, context_size, gpu_layers, extra_info=None):
        super().__init__()
        self.port = str(port)
        self.model_name = model_name
        self.info = {
            "port": self.port,
            "model": model_name,
            "ctx": context_size,
            "layers": gpu_layers,
        }
        
        # Merge extra info if provided (offload, est_spill, etc)
        if isinstance(extra_info, dict):
            self.info.update(extra_info)
        elif isinstance(extra_info, str):
            self.info["offload"] = extra_info
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.finished.connect(self._handle_finished)
        self.process.errorOccurred.connect(self._handle_error)
        
        # Stats Timer
        self._timer = QTimer(self)
        self._timer.setInterval(2000) # 2s interval
        self._timer.timeout.connect(self._update_stats)
        
        # Log Buffer
        self._buffer = ""
        self._log_queue = []
        
        # Log Flush Timer (Anti-Freeze)
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(100) # 100ms = 10fps updates
        self._flush_timer.timeout.connect(self._flush_logs)
        self._flush_timer.start()

    def start(self, cmd, env=None):
        self.process.setProgram(cmd[0])
        self.process.setArguments(cmd[1:])
        
        if env:
            self.process.setEnvironment(env)
            
        self.process.start()
        self._timer.start()
        self.status_changed.emit(self.port, True)
    
    def stop(self):
        self._timer.stop()
        if self.process.state() == QProcess.Running:
            self.process.kill() # Terminate immediately
            self.process.waitForFinished(2000) # Wait up to 2s for cleanup

    def _update_stats(self):
        if self.process.state() != QProcess.Running:
            return
            
        try:
            pid = self.process.processId()
            if pid <= 0: return
            
            import psutil
            p = psutil.Process(pid)
            mem_bytes = p.memory_info().rss
            mem_gb = mem_bytes / (1024 ** 3)
            
            stats = {
                "ram_gb": mem_gb,
                # cpu_percent could be added later
            }
            # Include confirmed offload status if available (so UI catches up if it missed the event)
            if "offload_confirmed" in self.info:
                stats["offload_confirmed"] = self.info["offload_confirmed"]
                
            self.stats_updated.emit(self.port, stats)
        except Exception:
            pass # Process might have died or psutil error

    def _handle_stdout(self):
        data = self.process.readAllStandardOutput()
        text = str(data, encoding='utf-8', errors='replace')
        
        # Add to buffer
        self._buffer += text
        
        # Process complete lines
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            line = line.strip()
            if not line: continue
            
            # Emit raw log line
            # self.log_received.emit(self.port, line) # DISABLE DIRECT EMIT
            self._log_queue.append(line)
            
            # Parse Offload Status
            # "llm_load_tensors: offloaded 33/33 layers to GPU"
            # Note: llama.cpp often outputs ANSI color codes, e.g. "offloaded \x1b[1m33/33\x1b[0m"
            
            # 1. Clean ANSI codes for parsing
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            
            if "offloaded" in clean_line.lower():
                # Emit debug to GUI so user sees it
                # self.log_received.emit(self.port, f"DEBUG: Found offload line: '{clean_line.strip()}'")
                
                try:
                    # Very Relaxed regex: look for "N/M" pattern
                    # Handle: "offloaded 33/33", "offloaded: 33/33", "offloaded \t 33/33"
                    match = re.search(r"offloaded.*?(\d+)\s*/\s*(\d+)", clean_line, re.IGNORECASE)
                    if match:
                        done, total = match.groups()
                        status = f"{done}/{total}"
                        self.info["offload_confirmed"] = status
                        self.stats_updated.emit(self.port, {"offload_confirmed": status})
                        # self.log_received.emit(self.port, f"DEBUG: Parsed status '{status}'")
                    else:
                        pass
                        # self.log_received.emit(self.port, f"DEBUG: Regex failed on '{clean_line.strip()}'")
                except Exception as e:
                    pass

            # Parse Buffer Sizes (Real Spillage)
            # "load_tensors:   CPU_Mapped model buffer size =   308.23 MiB"
            # "load_tensors:        CUDA0 model buffer size =  2513.90 MiB"
            if "model buffer size =" in clean_line:
                try:
                    # CPU Spillage
                    if "CPU_Mapped" in clean_line:
                        match = re.search(r"=\s+([\d\.]+)\s+MiB", clean_line)
                        if match:
                            mb = float(match.group(1))
                            self.info["cpu_buffer_mb"] = mb
                            self.stats_updated.emit(self.port, {"cpu_buffer_mb": mb})
                    
                    # GPU Usage
                    elif "CUDA" in clean_line:
                        match = re.search(r"=\s+([\d\.]+)\s+MiB", clean_line)
                        if match:
                            mb = float(match.group(1))
                            self.info["gpu_buffer_mb"] = mb
                            self.stats_updated.emit(self.port, {"gpu_buffer_mb": mb})
                except:
                    pass

            # Check for Errors
            if "error" in clean_line.lower() or "exception" in clean_line.lower():
                 self.stats_updated.emit(self.port, {"offload_confirmed": "Error"})

    def _handle_finished(self, exit_code, exit_status):
        self.status_changed.emit(self.port, False)
        self.finished.emit(self.port, exit_code)

    def _handle_error(self, error):
        # Maps QProcess::ProcessError enum to string if needed, 
        # but usually we just log that it crashed or failed to start
        if error == QProcess.FailedToStart:
             self.log_received.emit(self.port, "Process failed to start (check path/permissions).")
             self.status_changed.emit(self.port, False)

    def _flush_logs(self):
        """
        Emits buffered logs in a single signal to avoid overwhelming the event loop.
        """
        if not self._log_queue:
            return
            
        # Join all queued lines into one large string
        # This is much more efficient for Qt than hundreds of individual signals
        # We join with '' because the lines already contain their newlines (or we add them)
        # Actually in _handle_stdout we strip() then maybe we should re-add newlines?
        # The original code did: self.log_received.emit(self.port, line)
        # And MainWindow did: self.log_output.insertPlainText(f"[{port}] {msg}")
        # So we should join with newlines here if we want to emit one chunk.
        
        # However, Main Window usage: formatted = f"[{port}] {msg}"
        # If we send a multiline string, MainWindow will prepend [port] only to the first line?
        # Let's check MainWindow.handle_session_log:
        # formatted = f"[{port}] {msg}"
        # self.log_output.insertPlainText(formatted)
        
        # If 'msg' has newlines, insertPlainText inserts it as is.
        # But we want [port] prefix on every line? 
        # The original code emitted line-by-line, so every line got [port].
        # If we buffer, we lose that unless we format here or change MainWindow.
        # To preserve exact behavior without changing MainWindow, we can:
        # 1. Format here: `formatted_chunk = "\n".join([f"[{self.port}] {line}" for line in self._log_queue])`
        # 2. Emit `formatted_chunk` and have MainWindow just print it.
        # But MainWindow expects `(port, msg)`.
        
        # Compromise: Emit the raw chunk. MainWindow will put [port] at the start of the chunk.
        # This means subsequent lines in the chunk won't have [port]. 
        # For a high-speed log view, maybe that's acceptable? 
        # actually, `llama-server` output is often just progress bars. 
        # Let's try to maintain "one prefix per line" if possible, or just accept the improved performance.
        # Better: let's modify MainWindow to handle multiline messages if needed, 
        # OR just join with newlines and accept that [port] is only at the top of the batch.
        # Given the "freeze" is the priority, batching is key.
        
        # Let's prepend [port] here implies we change the signal contract or logic?
        # No, signal is (port, msg). 
        # If we emit (port, "line1\nline2"), MainWindow prints "[port] line1\nline2".
        # This is fine. The user knows which tab they are in.
        
        chunk = "\n".join(self._log_queue)
        self._log_queue.clear()
        self.log_received.emit(self.port, chunk + "\n") # Restore trailing newline for the batch

