from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QLineEdit, QTextEdit,
                               QProgressBar, QScrollArea, QFileDialog, QMessageBox, QFrame)
from PySide6.QtCore import Qt, QProcess, QTimer, Signal
from scripts.gui.widgets import Card, ResourceMeter
from scripts.quantize_merge import list_models, find_split_files
from scripts.gguf_parser import clear_model_cache
from scripts.config import save_config, CACHE_DIR
import os
import shutil
import sys

def calculate_overflow_state(stats: dict, is_fully_offloaded: bool = None) -> tuple[float, bool, str]:
    """
    Calculates VRAM overflow based on CPU buffer and System RAM.
    Returns: (overflow_gb, is_real, suffix_label)
    """
    cpu_mb = stats.get('cpu_buffer_mb', 0.0)
    ram_gb = stats.get('ram_gb', 0.0)
    
    # If is_fully_offloaded not provided, try to parse from stats
    if is_fully_offloaded is None:
        offload_str = stats.get('offload_confirmed', "")
        is_fully_offloaded = False
        if offload_str:
            try:
                done, total = map(int, offload_str.split('/'))
                if done == total and total > 0: is_fully_offloaded = True
            except: pass

    overflow_gb = 0.0
    is_real = False
    suffix = ""
    
    # 1. Check explicit log report first (CPU_Mapped)
    raw_mb = float(cpu_mb) if cpu_mb is not None else 0.0
    
    if raw_mb > 512:
        # Big explicit spill (Weights or KV Cache reported by llama.cpp)
        overflow_gb = raw_mb / 1024.0
        is_real = True
        suffix = " (Model)"
    elif is_fully_offloaded:
        # Layers are on GPU, but maybe KV cache spilled to RAM?
        # Check Process RAM. If > 1.0GB, assume spill.
        if ram_gb > 1.0:
            # Heuristic: RAM - Baseline Overhead (~0.5GB)
            overflow_gb = max(0.0, ram_gb - 0.5)
            is_real = True
            suffix = " (Sys RAM)"
        else:
            overflow_gb = 0.0 # Truly fits
    elif raw_mb > 0:
         # Partial offload partial spill
         overflow_gb = raw_mb / 1024.0
         is_real = True
         
    return overflow_gb, is_real, suffix

class HomePage(QWidget):
    request_tab_switch = Signal(int)
    request_stop_server = Signal(str) # port
    request_view_logs = Signal(str)   # port

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # Header
        header = QLabel("Welcome to Llama Manager")
        header.setStyleSheet("font-size: 28px; font-weight: bold; color: #0d6efd;")
        layout.addWidget(header, alignment=Qt.AlignCenter)
        
        sub_header = QLabel("Your all-in-one local LLM command center.")
        sub_header.setStyleSheet("font-size: 16px; color: #aaa;")
        layout.addWidget(sub_header, alignment=Qt.AlignCenter)
        
        layout.addSpacing(20)
        
        # Action Grid
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(20)
        
        # Launch Server Card
        self.create_action_card(grid_layout, "🚀 Launch Server", 
                                "Configure and start the llama.cpp server with full GPU control.",
                                1)
                                
        # Quantize Card
        self.create_action_card(grid_layout, "⚖️ Quantize Model", 
                                "Convert full-weight GGUF models to smaller quantized formats.",
                                2)
                                
        # Merge Card
        self.create_action_card(grid_layout, "🔗 Merge Models", 
                                "Combine split GGUF parts into a single usable file.",
                                3)

        # Settings Card
        self.create_action_card(grid_layout, "⚙️ Maintenance", 
                                "Reset defaults, clear caches, and manage system strings.",
                                4)
                                
        layout.addLayout(grid_layout)
        
        layout.addSpacing(30)
        
        # --- Active Sessions Section ---
        self.sessions_label = QLabel("Active Sessions")
        self.sessions_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #fff;")
        layout.addWidget(self.sessions_label)
        
        self.sessions_container = QWidget()
        self.sessions_layout = QVBoxLayout(self.sessions_container)
        self.sessions_layout.setContentsMargins(0, 0, 0, 0)
        self.sessions_layout.setSpacing(10)
        layout.addWidget(self.sessions_container)
        
        # Initial Empty State
        self.render_empty_state()
        
        self.cards = {} # { port_str: widget }
        
        layout.addStretch()
        
        # Footer
        footer = QLabel("System Status: Ready")
        footer.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(footer, alignment=Qt.AlignCenter)

    def create_action_card(self, layout, title, desc, tab_index):
        card = QFrame()
        card.setStyleSheet("""
            QFrame { 
                background-color: #2b2b2b; 
                border-radius: 10px; 
                border: 1px solid #3d3d3d;
            }
            QFrame:hover {
                background-color: #333333;
                border: 1px solid #0d6efd;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #fff; border: none; background: transparent;")
        card_layout.addWidget(lbl_title)
        
        lbl_desc = QLabel(desc)
        lbl_desc.setStyleSheet("color: #ccc; font-size: 13px; border: none; background: transparent;")
        lbl_desc.setWordWrap(True)
        card_layout.addWidget(lbl_desc)
        
        btn = QPushButton("Go")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("background-color: #0d6efd; color: white; border: none; padding: 8px; border-radius: 4px;")
        btn.clicked.connect(lambda: self.request_tab_switch.emit(tab_index))
        card_layout.addWidget(btn)
        
        layout.addWidget(card)

    def render_empty_state(self):
        self.clear_sessions()
        lbl = QLabel("No active LLM sessions running.")
        lbl.setStyleSheet("color: #666; font-style: italic; padding: 20px; border: 1px dashed #444; border-radius: 6px;")
        lbl.setAlignment(Qt.AlignCenter)
        self.sessions_layout.addWidget(lbl)

    def clear_sessions(self):
        while self.sessions_layout.count():
            item = self.sessions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def update_session_status(self, is_running: bool, info: dict = None):
        # Legacy method kept for compatibility, but redirects to new logic
        if is_running and info:
            self.add_session_card(info)
        elif not is_running and info:
            self.remove_session_card(info.get('port'))
        # If is_running=False and no info, we might want to clear all? 
        # But MainWindow handles specific logic now.

    def add_session_card(self, info: dict):
        if self.sessions_layout.count() == 1:
             # Check if it's the empty state label
             item = self.sessions_layout.itemAt(0)
             if isinstance(item.widget(), QLabel) and item.widget().text() == "No active LLM sessions running.":
                 self.clear_sessions()

        # Check if card already exists for this port?
        # Check if card already exists
        port = str(info.get('port'))
        if port in self.cards:
             return # Already exists

        self.render_active_session(info)

    def remove_session_card(self, port):
        port = str(port)
        if port in self.cards:
            w = self.cards[port]
            self.sessions_layout.removeWidget(w)
            w.deleteLater()
            del self.cards[port]
        
        if self.sessions_layout.count() == 0:
            self.render_empty_state()

    def render_active_session(self, info: dict):
        # Card Container
        card = QFrame()
        card.setProperty("port", str(info.get('port')))
        card.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #198754;
                border-radius: 8px;
            }
        """)
        
        # Main Layout (Vertical)
        main_layout = QVBoxLayout(card)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # --- Row 1: Header (Icon + Model + Buttons) ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        status_icon = QLabel("🟢")
        status_icon.setStyleSheet("font-size: 16px; border: none; background: transparent;")
        header_layout.addWidget(status_icon)
        
        model_name = info.get("model", "Unknown Model")
        model_lbl = QLabel(model_name)
        model_lbl.setStyleSheet("font-weight: bold; font-size: 16px; color: #fff; border: none; background: transparent;")
        header_layout.addWidget(model_lbl)
        
        header_layout.addStretch()
        
        # Buttons
        btn_logs = QPushButton("Terminal / Logs")
        btn_logs.setCursor(Qt.PointingHandCursor)
        btn_logs.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #444; border-color: #888; }
        """)
        btn_logs.clicked.connect(lambda: self.request_view_logs.emit(info.get('port')))
        header_layout.addWidget(btn_logs)
        
        btn_stop = QPushButton("Stop Instance")
        btn_stop.setCursor(Qt.PointingHandCursor)
        btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #bb2d3b; }
        """)
        # Custom handler for immediate feedback
        def on_stop_click():
            btn_stop.setEnabled(False)
            btn_stop.setText("Stopping...")
            btn_stop.setStyleSheet("background-color: #666; color: #aaa; border: none; padding: 6px 12px; border-radius: 4px;")
            self.request_stop_server.emit(info.get('port'))

        btn_stop.clicked.connect(on_stop_click)
        header_layout.addWidget(btn_stop)
        
        main_layout.addLayout(header_layout)
        
        # --- Row 2: Meta Info ---
        meta_layout = QHBoxLayout()
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(10)
        
        # Base Info
        meta_text = f"Port: {info.get('port', '8080')}   |   Context: {info.get('ctx', '?')}   |   GPU Layers: {info.get('layers', '?')}"
        meta_lbl = QLabel(meta_text)
        meta_lbl.setStyleSheet("color: #aaa; font-size: 13px; border: 0px; background: transparent;")
        meta_layout.addWidget(meta_lbl)
        
        # Spacing (No Divider Widget)
        meta_layout.addSpacing(20)

        # Offload Status
        # Default text
        card.offload_lbl = QLabel("GPU: Pending...")
        card.offload_lbl.setStyleSheet("color: #aaa; font-size: 13px; font-weight: bold; border: 0px; background: transparent;")
        
        if "offload_confirmed" in info:
             status = info["offload_confirmed"]
             card.offload_lbl.setText(f"GPU: {status} Layers")
             # Determine color based on matching numbers
             try:
                 done, total = map(int, status.split('/'))
                 if done == total and total > 0:
                     card.offload_lbl.setStyleSheet("color: #198754; font-weight: bold; font-size: 13px; border: 0px; background: transparent;") 
                 elif done == 0:
                     card.offload_lbl.setStyleSheet("color: #dc3545; font-weight: bold; font-size: 13px; border: 0px; background: transparent;") 
                 else:
                     card.offload_lbl.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 13px; border: 0px; background: transparent;") 
             except:
                 pass

        meta_layout.addWidget(card.offload_lbl)

        # Spacing
        meta_layout.addSpacing(20)
        
        # ... (Next steps handled in previous edit) ...

        # RAM Usage
        card.ram_lbl = QLabel("System RAM (Process): Waiting...")
        card.ram_lbl.setStyleSheet("color: #aaa; font-size: 13px; border: 0px; background: transparent;")
        meta_layout.addWidget(card.ram_lbl)
        
        # Spill Label (No Widget Divider)
        card.spill_lbl = QLabel("")
        card.spill_lbl.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 13px; border: 0px; background: transparent; margin-left: 20px;")
        meta_layout.addWidget(card.spill_lbl)
        
        # Move stretch to the END to prevent gaps
        meta_layout.addStretch()
        main_layout.addLayout(meta_layout)
        
        # Initial State: Calculate Overflow
        est_spill = float(info.get("est_spill_gb", 0))
        real_spill_mb = info.get("cpu_buffer_mb", None)
        offload_str = info.get("offload_confirmed", "")
        
        overflow_gb = est_spill
        is_real = False
        
        # 1. If 100% offloaded, Overflow is 0.0 (unless huge spill)
        raw_mb = float(real_spill_mb) if real_spill_mb is not None else 0.0
        
        is_fully_offloaded = False
        if offload_str:
            try:
                done, total = map(int, offload_str.split('/'))
                if done == total and total > 0:
                    is_fully_offloaded = True
            except:
                pass

        if is_fully_offloaded and raw_mb < 512:
            overflow_gb = 0.0
            is_real = True
        elif raw_mb > 0:
            overflow_gb = raw_mb / 1024.0
            is_real = True

        txt = f"VRAM Overflow: {overflow_gb:.2f} GB"
        if is_real: txt += " (Actual)"
        else: txt += " (Est)"
        card.spill_lbl.setText(txt)
        
        self.sessions_layout.addWidget(card)
        self.cards[str(info.get("port"))] = card
        
    def update_session_stats(self, port, stats):
        port = str(port)
        if port in self.cards:
            card = self.cards[port]
            
            # Update RAM
            if 'ram_gb' in stats and hasattr(card, 'ram_lbl'):
                gb = stats['ram_gb']
                card.ram_lbl.setText(f"System RAM (Process): {gb:.2f} GB")
                card.ram_lbl.setStyleSheet("color: #0d6efd; font-size: 13px; font-weight: bold; border: none; background: transparent;")
                
            # Update Offload
            if 'offload_confirmed' in stats and hasattr(card, 'offload_lbl'):
                status = stats['offload_confirmed']
                card.offload_lbl.setText(f"GPU: {status} Layers")
                try:
                    done, total = map(int, status.split('/'))
                    if done == total and total > 0:
                        card.offload_lbl.setStyleSheet("color: #198754; font-weight: bold; font-size: 13px; border: none; background: transparent;") 
                    elif done == 0:
                        card.offload_lbl.setStyleSheet("color: #dc3545; font-weight: bold; font-size: 13px; border: none; background: transparent;")
                    else:
                        card.offload_lbl.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 13px; border: none; background: transparent;")
                except:
                    pass
            
            # Update Real Spill / Overflow (Using centralized logic)
            overflow_gb, is_real, suffix = calculate_overflow_state(stats)
            
            # Decide visibility based on overflow
            should_update = True
            
            if should_update and hasattr(card, 'spill_lbl'):
                 txt = f"VRAM Overflow: {overflow_gb:.2f} GB"
                 if suffix: txt += suffix
                 elif is_real: txt += " (Actual)"
                 else: txt += " (Est)"
                 
                 card.spill_lbl.setText(txt)
                 card.spill_lbl.setVisible(True)
                 
                 # Restore divider visibility management
                 if hasattr(card, 'spill_div'):
                     card.spill_div.setVisible(True)
                     card.spill_div.setStyleSheet("color: #444; font-weight: bold; font-size: 14px; border: 0px; background: transparent;")
                     
                 card.spill_lbl.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 13px; border: 0px; background: transparent;")

class QuantizePage(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.process = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("Quantize Model")
        header.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(header)
        
        # Form
        form_card = Card("Configuration")
        
        # Source Model
        self.model_combo = QComboBox()
        self.refresh_models()
        self.model_combo.currentIndexChanged.connect(self.suggest_output_name)
        form_card.layout.addWidget(QLabel("Source Model:"))
        form_card.layout.addWidget(self.model_combo)
        
        # Quant Type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q4_k_m", "q5_k_m", "q6_k"])
        self.type_combo.currentTextChanged.connect(self.suggest_output_name)
        form_card.layout.addWidget(QLabel("Quantization Type:"))
        form_card.layout.addWidget(self.type_combo)
        
        # Output Name
        self.output_input = QLineEdit()
        form_card.layout.addWidget(QLabel("Output Filename:"))
        form_card.layout.addWidget(self.output_input)
        
        layout.addWidget(form_card)
        
        # Actions
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Start Quantization")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #0d6efd; font-size: 16px;")
        self.btn_run.clicked.connect(self.run_quantize)
        btn_layout.addWidget(self.btn_run)
        layout.addLayout(btn_layout)
        
        # Logs
        layout.addWidget(QLabel("Process Log:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("font-family: Consolas; font-size: 12px; background: #111; color: #ddd;")
        layout.addWidget(self.log_output)
    
    def refresh_models(self):
        self.model_combo.clear()
        models = list_models(self.cfg)
        self.model_combo.addItems(models)
    
    def suggest_output_name(self):
        model = self.model_combo.currentText()
        if not model: return
        q_type = self.type_combo.currentText()
        base, ext = os.path.splitext(model)
        self.output_input.setText(f"{base}-{q_type}{ext}")

    def run_quantize(self):
        model = self.model_combo.currentText()
        q_type = self.type_combo.currentText()
        out_name = self.output_input.text()
        
        if not model or not out_name:
            QMessageBox.warning(self, "Error", "Please select a model and output name.")
            return

        qbin = os.path.join(self.cfg["bin_path"], "llama-quantize")
        if os.name == "nt": qbin += ".exe"
            
        src_path = os.path.join(self.cfg["model_path"], model)
        out_path = os.path.join(self.cfg["model_path"], out_name)
        
        self.log_output.clear()
        self.log_output.append(f"> Quantizing {model} to {q_type}...")
        self.log_output.append(f"> Output: {out_path}\n")
        
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        self.process.setProgram(qbin)
        self.process.setArguments([src_path, out_path, q_type])
        self.process.start()
        
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running...")

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        text = str(data, encoding='utf-8', errors='replace')
        self.log_output.moveCursor(self.log_output.textCursor().End)
        self.log_output.insertPlainText(text)
        self.log_output.ensureCursorVisible()
        
    def process_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Start Quantization")
        if self.process.exitStatus() == QProcess.NormalExit and self.process.exitCode() == 0:
            self.log_output.append("\n> Quantization Complete Successfully!")
            QMessageBox.information(self, "Success", "Quantization complete!")
            self.refresh_models() # Update list to include new file
        else:
            self.log_output.append("\n> Process failed.")

class MergePage(QWidget):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.process = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
        header = QLabel("Merge Split Models")
        header.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(header)
        
        # Form
        form_card = Card("Configuration")
        
        self.split_combo = QComboBox()
        self.refresh_splits()
        self.split_combo.currentIndexChanged.connect(self.suggest_output_name)
        form_card.layout.addWidget(QLabel("First Split Part:"))
        form_card.layout.addWidget(self.split_combo)
        
        self.output_input = QLineEdit()
        form_card.layout.addWidget(QLabel("Output Filename:"))
        form_card.layout.addWidget(self.output_input)
        
        layout.addWidget(form_card)
        
         # Actions
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Merge Files")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #0d6efd; font-size: 16px;")
        self.btn_run.clicked.connect(self.run_merge)
        btn_layout.addWidget(self.btn_run)
        layout.addLayout(btn_layout)
        
        # Logs
        layout.addWidget(QLabel("Process Log:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("font-family: Consolas; font-size: 12px; background: #111; color: #ddd;")
        layout.addWidget(self.log_output)

    def refresh_splits(self):
        self.split_combo.clear()
        files = find_split_files(self.cfg)
        display_names = [os.path.basename(f) for f in files]
        self.split_combo.addItems(display_names)
        
    def suggest_output_name(self):
        current = self.split_combo.currentText()
        if not current: return
        # Simple heuristic: remove -00001-of-XXXXX or similar
        import re
        new_name = re.sub(r'-\d+-of-\d+', '', current)
        new_name = re.sub(r'\.\d+$', '', new_name) # .gguf.00001 -> .gguf
        if not new_name.endswith('.gguf'):
            new_name += '.gguf'
        self.output_input.setText("merged-" + new_name)

    def run_merge(self):
        fname = self.split_combo.currentText()
        out_name = self.output_input.text()
        
        if not fname or not out_name:
            QMessageBox.warning(self, "Error", "Please select a file and output name.")
            return

        # Need full path for source
        # find_split_files returns full paths, but we populated combo with basenames
        # Re-fetch logic or search
        full_path = None
        current_splits = find_split_files(self.cfg)
        for p in current_splits:
            if os.path.basename(p) == fname:
                full_path = p
                break
        
        if not full_path:
             QMessageBox.critical(self, "Error", "Could not resolve full path")
             return

        split_bin = os.path.join(self.cfg["bin_path"], "llama-gguf-split")
        if os.name == "nt": split_bin += ".exe"
            
        out_path = os.path.join(self.cfg["model_path"], out_name)
        
        self.log_output.clear()
        self.log_output.append(f"> Merging {fname}...")
        self.log_output.append(f"> Output: {out_path}\n")
        
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        self.process.setProgram(split_bin)
        self.process.setArguments(["--merge", full_path, out_path])
        self.process.start()
        
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Merging...")

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        text = str(data, encoding='utf-8', errors='replace')
        self.log_output.moveCursor(self.log_output.textCursor().End)
        self.log_output.insertPlainText(text)
        self.log_output.ensureCursorVisible()
        
    def process_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Merge Files")
        if self.process.exitStatus() == QProcess.NormalExit and self.process.exitCode() == 0:
            self.log_output.append("\n> Merge Complete Successfully!")
            QMessageBox.information(self, "Success", "Merge complete!")
        else:
            self.log_output.append("\n> Process failed.")

class SettingsPage(QWidget):
    paths_saved = Signal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        header = QLabel("Maintenance & Settings")
        header.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(header)

        # --- Paths Configuration (must be first) ---
        paths_card = Card("Paths Configuration")

        paths_card.layout.addWidget(QLabel("llama-server binary directory:"))
        bin_row = QHBoxLayout()
        self.bin_path_input = QLineEdit(cfg.get("bin_path", ""))
        self.bin_path_input.setPlaceholderText("/path/to/llama.cpp/build/bin")
        bin_row.addWidget(self.bin_path_input)
        btn_browse_bin = QPushButton("Browse…")
        btn_browse_bin.setFixedWidth(90)
        btn_browse_bin.clicked.connect(self._browse_bin)
        bin_row.addWidget(btn_browse_bin)
        paths_card.layout.addLayout(bin_row)

        paths_card.layout.addWidget(QLabel("Model directory (.gguf files):"))
        model_row = QHBoxLayout()
        self.model_path_input = QLineEdit(cfg.get("model_path", ""))
        self.model_path_input.setPlaceholderText("/path/to/LLMs")
        model_row.addWidget(self.model_path_input)
        btn_browse_model = QPushButton("Browse…")
        btn_browse_model.setFixedWidth(90)
        btn_browse_model.clicked.connect(self._browse_model)
        model_row.addWidget(btn_browse_model)
        paths_card.layout.addLayout(model_row)

        btn_save = QPushButton("Save Paths")
        btn_save.setStyleSheet(
            "QPushButton { background-color: #198754; color: white; font-weight: bold; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #157347; }"
        )
        btn_save.clicked.connect(self._save_paths)
        paths_card.layout.addWidget(btn_save)

        layout.addWidget(paths_card)
        
        # --- Model Defaults Section ---
        defaults_card = Card("Model Defaults")
        defaults_card.layout.addWidget(QLabel("Reset saved settings (Context Size, GPU Layers, etc.) for specific models."))
        
        hbox_defaults = QHBoxLayout()
        self.model_combo = QComboBox()
        self.refresh_models()
        self.model_combo.setMinimumWidth(300)
        hbox_defaults.addWidget(self.model_combo)
        
        btn_reset_one = QPushButton("Reset Selected")
        btn_reset_one.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold; padding: 5px;")
        btn_reset_one.clicked.connect(self.reset_selected_defaults)
        hbox_defaults.addWidget(btn_reset_one)
        
        btn_reset_all = QPushButton("Reset ALL Models")
        btn_reset_all.setStyleSheet("background-color: #dc3545; font-weight: bold; padding: 5px;")
        btn_reset_all.clicked.connect(self.reset_all_defaults)
        hbox_defaults.addWidget(btn_reset_all)
        
        defaults_card.layout.addLayout(hbox_defaults)
        layout.addWidget(defaults_card)
        
        # --- Cache Management Section ---
        cache_card = Card("Cache Management")
        
        # Model Arch Cache
        hbox_cache_arch = QHBoxLayout()
        lbl_arch = QLabel("Model Architecture Cache:")
        lbl_arch.setStyleSheet("font-weight: bold;")
        hbox_cache_arch.addWidget(lbl_arch)
        
        btn_clear_arch_cache = QPushButton("Clear Architecture Cache")
        btn_clear_arch_cache.setStyleSheet("padding: 5px;")
        btn_clear_arch_cache.clicked.connect(self.clear_arch_cache)
        hbox_cache_arch.addWidget(btn_clear_arch_cache)
        cache_card.layout.addLayout(hbox_cache_arch)
        cache_card.layout.addWidget(QLabel("Clears cached GGUF metadata (layer counts, tensor types). Useful if models are updated."))

        # System/GPU Cache
        hbox_cache_sys = QHBoxLayout()
        lbl_sys = QLabel("System Hardware Cache:")
        lbl_sys.setStyleSheet("font-weight: bold;")
        hbox_cache_sys.addWidget(lbl_sys)
        
        btn_clear_sys_cache = QPushButton("Clear System Cache")
        btn_clear_sys_cache.setStyleSheet("padding: 5px;")
        btn_clear_sys_cache.clicked.connect(self.clear_system_cache)
        hbox_cache_sys.addWidget(btn_clear_sys_cache)
        cache_card.layout.addLayout(hbox_cache_sys)
        cache_card.layout.addWidget(QLabel("Clears cached GPU capabilities and binary flags. Use if you changed hardware or drivers."))

        layout.addWidget(cache_card)
        
        layout.addStretch()

    # --- Path helpers ---

    def _browse_bin(self):
        start = self.bin_path_input.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "Select binary directory", start)
        if d:
            self.bin_path_input.setText(d)

    def _browse_model(self):
        start = self.model_path_input.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "Select model directory", start)
        if d:
            self.model_path_input.setText(d)

    def _save_paths(self):
        bin_path = self.bin_path_input.text().strip()
        model_path = self.model_path_input.text().strip()

        if not bin_path or not model_path:
            QMessageBox.warning(self, "Missing paths", "Both paths are required.")
            return

        # Validate directories exist
        missing = [p for p in (bin_path, model_path) if not os.path.isdir(p)]
        if missing:
            QMessageBox.warning(self, "Path not found",
                                "These directories don't exist:\n" + "\n".join(missing))
            return

        # Validate llama-server binary is present
        exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        if not os.path.isfile(os.path.join(bin_path, exe)):
            resp = QMessageBox.question(
                self, "Binary not found",
                f"{exe} was not found in the selected directory.\nSave anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if resp == QMessageBox.No:
                return

        self.cfg["bin_path"] = bin_path
        self.cfg["model_path"] = model_path
        save_config(self.cfg)
        self.refresh_models()
        QMessageBox.information(self, "Saved", "Paths saved. Model list refreshed.")
        self.paths_saved.emit()

    # --- Model helpers ---

    def refresh_models(self):
        self.model_combo.clear()
        models = list_models(self.cfg)
        self.model_combo.addItems(models)

    def reset_selected_defaults(self):
        model = self.model_combo.currentText()
        if not model: return
        
        current_defaults = self.cfg.get("model_defaults", {})
        if model in current_defaults:
            del self.cfg["model_defaults"][model]
            save_config(self.cfg)
            QMessageBox.information(self, "Success", f"Defaults reset for {model}")
        else:
            QMessageBox.information(self, "Info", f"No saved defaults found for {model}")

    def reset_all_defaults(self):
        ret = QMessageBox.question(self, "Confirm", "Are you sure you want to reset defaults for ALL models?",
                                   QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            self.cfg["model_defaults"] = {}
            save_config(self.cfg)
            QMessageBox.information(self, "Success", "All model defaults reset.")

    def clear_arch_cache(self):
        try:
            clear_model_cache() # Clears all
            QMessageBox.information(self, "Success", "Model Architecture Cache Cleared.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clear cache: {e}")

    def clear_system_cache(self):
        # Delete *_system_flags.json and *_flags.json in CACHE_DIR
        try:
            count = 0
            if CACHE_DIR.exists():
                for f in CACHE_DIR.glob("*_flags.json"):
                    f.unlink()
                    count += 1
            QMessageBox.information(self, "Success", f"Cleared {count} system cache files.\nRestart dashboard to re-detect hardware.")
        except Exception as e:
             QMessageBox.critical(self, "Error", f"Failed to clear system cache: {e}")
