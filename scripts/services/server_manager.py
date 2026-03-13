from PySide6.QtCore import QObject, Signal, QProcess
from scripts.gui.workers import ServerSession
import socket

class ServerManager(QObject):
    """
    Central service for managing llama-server instances.
    Decouples process management from UI logic (MCCC Compliance).
    """
    # Signals to UI
    session_started = Signal(dict) # info dict
    session_stopped = Signal(str)  # port
    log_received = Signal(str, str) # port, message
    error_occurred = Signal(str, str) # port, error message
    stats_updated = Signal(str, dict) # port, stats

    def __init__(self):
        super().__init__()
        self.sessions = {} # { str(port): ServerSession }

    def launch_server(self, port, model_name, ctx_size, gpu_layers, cmd, env=None, extra_info=None):
        """
        Creates and starts a new ServerSession.
        """
        port = str(port)
        
        # Stop existing if any (though UI should prevent this)
        if port in self.sessions:
            self.stop_server(port)
            
        session = ServerSession(port, model_name, ctx_size, gpu_layers, extra_info)
        
        # Connect signals
        session.log_received.connect(self._on_log)
        session.status_changed.connect(self._on_status)
        session.finished.connect(self._on_finished)
        session.stats_updated.connect(self._on_stats_updated)
        
        self.sessions[port] = session
        session.start(cmd, env)
        
        # Emit start immediately for UI feedback
        # (Real status confirmation comes via _on_status)
        self.session_started.emit(session.info)
        return session

    def stop_server(self, port):
        port = str(port)
        if port in self.sessions:
            self.sessions[port].stop()

    def stop_all(self):
        for port in list(self.sessions.keys()):
            self.stop_server(port)

    def get_active_sessions(self):
        return [s.info for s in self.sessions.values()]

    def find_free_port(self, start_port):
        port = start_port
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
                port += 1

    # --- Internal Handlers ---

    def _on_log(self, port, msg):
        self.log_received.emit(port, msg)

    def _on_stats_updated(self, port, stats):
        self.stats_updated.emit(port, stats)

    def _on_status(self, port, is_running):
        # We could relay this, or let the UI handle it via session object
        # For MCCC, let's keep it clean: strictly business logic here?
        # Actually UI needs to know when to remove the card.
        pass

    def _on_finished(self, port, exit_code):
        self.session_stopped.emit(port)
        if port in self.sessions:
            del self.sessions[port]
