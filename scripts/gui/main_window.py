import sys
import os
import subprocess
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QComboBox, QSlider, QScrollArea,
                               QGridLayout, QGroupBox, QCheckBox, QMessageBox, QTabWidget, QTextEdit, QApplication, QLineEdit)
from PySide6.QtCore import Qt, QTimer, QProcess, QThread, Signal
from PySide6.QtGui import QTextCursor, QIntValidator

# Import existing logic
from scripts.config import load_config, save_config
from scripts.quantize_merge import list_models
from scripts.detectors import probe_gpu_support, detect_model_family
from scripts.gguf_parser import find_projector_file, get_model_architecture, list_projector_candidates
from scripts.vram_calculator import calculate_vram_usage, get_vram_available
from scripts.tui import get_model_capabilities, get_safe_defaults
from scripts.tui import get_model_capabilities, get_safe_defaults
from scripts.services.command_builder import ServerCommandBuilder, VisionConfig
from scripts.gui.widgets import VramBar, Card
from scripts.gui.styles import get_stylesheet
from scripts.gui.pages import HomePage, QuantizePage, MergePage, SettingsPage
from scripts.services.server_manager import ServerManager
from scripts.gui.widgets.vision_settings import VisionSettingsWidget
import socket

class MainWindow(QMainWindow):
    # Fixed Signature: port, model, ctx, layers, cmd(list), env(list), info(dict)
    request_launch = Signal(int, str, int, str, list, list, dict) 
    request_stop_server = Signal(str)
    request_stop_all = Signal()

    def closeEvent(self, event):
        """
        Ensure clean shutdown of background threads and processes.
        """
        # 1. Request all servers to stop
        self.request_stop_all.emit()
        
        # 2. Stop the Manager Thread
        if self.manager_thread.isRunning():
            self.manager_thread.quit()
            # Wait a bit for graceful cleanup
            if not self.manager_thread.wait(3000):
                self.manager_thread.terminate()
                
        event.accept()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Llama Manager Dashboard")
        self.resize(1100, 750)
        
        self.cfg = load_config()
        self.models = list_models(self.cfg)
        self.current_model = None
        self.current_projector = None
        
        # Multi-Instance State
        # Multi-Instance State
        self.server_manager = ServerManager()
        self.manager_thread = QThread()
        self.server_manager.moveToThread(self.manager_thread)
        
        self.server_manager.log_received.connect(self.handle_session_log)
        self.server_manager.session_stopped.connect(self.handle_session_stopped)
        self.server_manager.session_started.connect(self.handle_session_started)
        self.server_manager.stats_updated.connect(self.handle_session_stats)
        
        # Connect Requests (Cross-Thread)
        self.request_launch.connect(self.server_manager.launch_server)
        self.request_stop_server.connect(self.server_manager.stop_server)
        self.request_stop_all.connect(self.server_manager.stop_all)
        
        self.manager_thread.start()
        
        self.next_port = int(self.cfg['defaults'].get('port', 8080))
        
        self.gpu_checks = []
        
        # UI Setup
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QHBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Sidebar
        self.create_sidebar()
        
        # Content Area
        self.content_stack = QTabWidget()
        self.content_stack.tabBar().setVisible(False) # Hide tabs, use sidebar nav
        self.content_stack.setStyleSheet("background-color: #1e1e1e;")
        self.main_layout.addWidget(self.content_stack)
        
        # Pages
        self.page_home = HomePage()
        self.page_home.request_tab_switch.connect(lambda idx: self.switch_tab(idx, self.nav_btns[idx]))
        # Connect Home Page Signals
        self.page_home.request_stop_server.connect(self.stop_server_by_port)
        self.page_home.request_view_logs.connect(lambda port: self.switch_to_logs(port))
        self.content_stack.addTab(self.page_home, "Home")

        self.page_launch = QWidget()
        self.setup_launch_page()
        self.content_stack.addTab(self.page_launch, "Launch")
        
        self.page_quant = QuantizePage(self.cfg)
        self.content_stack.addTab(self.page_quant, "Quantize")

        self.page_merge = MergePage(self.cfg)
        self.content_stack.addTab(self.page_merge, "Merge")

        self.page_settings = SettingsPage(self.cfg)
        self.page_settings.paths_saved.connect(self._on_paths_saved)
        self.content_stack.addTab(self.page_settings, "Settings")
        
        self.page_logs = QWidget()
        self.setup_logs_page()
        self.content_stack.addTab(self.page_logs, "Logs")
        
        # Initialize with Home Page
        self.switch_tab(0, self.nav_btns[0])
        self.nav_btns[0].setChecked(True)

        # Status Bar (Global)
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("background-color: #252525; color: #aaa; border-top: 1px solid #3d3d3d;")
        
        self.status_lbl = QLabel("System Ready")
        self.status_lbl.setStyleSheet("color: #888; font-size: 12px; padding: 0 10px;")
        self.status_bar.addWidget(self.status_lbl)
        self.status_bar.show()

        # Initialize launch page models
        if self.models:
            # Don't auto-select to avoid parsing repeatedly at startup
            # User must select a model manually
            self.model_combo.setCurrentIndex(-1)
            # self.on_model_changed(self.model_combo.currentText())

    def create_sidebar(self):
        self.sidebar = QWidget()
        sidebar = self.sidebar # Alias for less typing locally
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background-color: #252525; border-right: 1px solid #3d3d3d;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 30, 20, 20)
        layout.setSpacing(15)
        
        title = QLabel("Llama Manager")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #0d6efd; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Nav Buttons
        self.nav_btns = []
        
        labels = ["Home", "Launch", "Quantize", "Merge", "Settings", "Logs"]
        for i, lbl in enumerate(labels):
            btn = QPushButton(lbl)
            btn.setCheckable(True)
            btn.setFixedHeight(40)
            btn.setStyleSheet("""
                QPushButton { text-align: left; padding-left: 15px; border: none; font-size: 14px; }
                QPushButton:checked { background-color: #3d3d3d; border-left: 3px solid #0d6efd; }
                QPushButton:hover { background-color: #333; }
            """)
            btn.clicked.connect(lambda checked, idx=i, b=btn: self.switch_tab(idx, b))
            layout.addWidget(btn)
            self.nav_btns.append(btn)
        
        layout.addStretch()
        
        # Status footer removed from here (moved to main bottom layout)
        # self.status_lbl = QLabel("Ready") ...
        
        self.main_layout.addWidget(sidebar)

    def switch_tab(self, index, active_btn):
        self.content_stack.setCurrentIndex(index)
        for btn in self.nav_btns:
            btn.setChecked(False)
            if btn == active_btn:
                btn.setChecked(True)
                # Apply active style manually if needed, or rely on QSS :checked

    def setup_logs_page(self):
        layout = QVBoxLayout(self.page_logs)
        layout.setContentsMargins(30, 30, 30, 30)
        
        label = QLabel("Server Output")
        label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #111; color: #0f0; font-family: Consolas, monospace; font-size: 13px; border: 1px solid #333; border-radius: 4px;")
        layout.addWidget(self.log_output)
        
        self.log_output.setStyleSheet("background-color: #111; color: #0f0; font-family: Consolas, monospace; font-size: 13px; border: 1px solid #333; border-radius: 4px;")
        layout.addWidget(self.log_output)
        
        # We remove the single stop button from logs page as we have multiple servers now.
        # Or we could have a "Stop All" button?
        btn_stop = QPushButton("Stop All Servers")
        btn_stop.setStyleSheet("""
            QPushButton { background-color: #dc3545; }
            QPushButton:hover { background-color: #bb2d3b; }
        """)
        btn_stop.setFixedWidth(150)
        btn_stop.clicked.connect(self.stop_all_servers)
        layout.addWidget(btn_stop, alignment=Qt.AlignRight)

    def setup_launch_page(self):
        layout = QVBoxLayout(self.page_launch)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Scroll Area for smaller screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0,0,0,0)
        scroll_layout.setSpacing(20)
        
        # Model Selection Card
        top_card = Card("Model Selection")
        
        # Horizontal layout for model combo + refresh button
        model_controls = QHBoxLayout()
        model_controls.setSpacing(10)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.models)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_controls.addWidget(self.model_combo)
        
        # Refresh button for models
        self.model_refresh_btn = QPushButton("🔄 Refresh")
        self.model_refresh_btn.setFixedWidth(100)
        self.model_refresh_btn.setToolTip("Reload model list from disk")
        self.model_refresh_btn.setStyleSheet("""
            QPushButton { 
                background-color: #3d3d3d; 
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
        """)
        self.model_refresh_btn.clicked.connect(self.refresh_model_list)
        model_controls.addWidget(self.model_refresh_btn)
        
        top_card.add_layout(model_controls)
        
        self.model_info_lbl = QLabel("Select a model...")
        self.model_info_lbl.setStyleSheet("color: #aaa; margin-top: 5px;")
        top_card.add_widget(self.model_info_lbl)
        
        # Projector Selection (Hidden by default)
        self.projector_container = QWidget()
        proj_layout = QVBoxLayout(self.projector_container)
        proj_layout.setContentsMargins(0, 10, 0, 0)
        proj_layout.setSpacing(5)
        
        proj_layout.addWidget(QLabel("Multimedia Projector (Vision):"))
        
        # Horizontal layout for combo + refresh button
        proj_controls = QHBoxLayout()
        proj_controls.setSpacing(10)
        
        self.projector_combo = QComboBox()
        self.projector_combo.currentTextChanged.connect(self.on_projector_changed)
        proj_controls.addWidget(self.projector_combo)
        
        # Refresh button
        self.projector_refresh_btn = QPushButton("🔄 Refresh")
        self.projector_refresh_btn.setFixedWidth(100)
        self.projector_refresh_btn.setToolTip("Reload projector files from disk")
        self.projector_refresh_btn.setStyleSheet("""
            QPushButton { 
                background-color: #3d3d3d; 
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
        """)
        self.projector_refresh_btn.clicked.connect(self.refresh_projector_list)
        proj_controls.addWidget(self.projector_refresh_btn)
        
        proj_layout.addLayout(proj_controls)

        top_card.add_widget(self.projector_container)
        self.projector_container.hide() # Hide unless vision model
        
        scroll_layout.addWidget(top_card)
        
        # Settings Group
        settings_group = QGroupBox("Configuration")
        grid = QGridLayout(settings_group)
        grid.setVerticalSpacing(15)
        grid.setHorizontalSpacing(20)
        
        # Context Size
        grid.addWidget(QLabel("Context Size:"), 0, 0)
        self.ctx_slider = QSlider(Qt.Horizontal)
        self.ctx_slider.setRange(512, 32768)
        self.ctx_slider.setSingleStep(512)
        self.ctx_slider.setSingleStep(512)
        self.ctx_slider.valueChanged.connect(self.update_ctx_input)
        self.ctx_slider.valueChanged.connect(self.update_vram_display)
        self.ctx_input = QLineEdit("2048")
        self.ctx_input.setFixedWidth(60)
        self.ctx_input.setAlignment(Qt.AlignCenter)
        self.ctx_input.setValidator(QIntValidator(512, 131072, self))
        self.ctx_input.editingFinished.connect(self.on_ctx_input_changed)
        grid.addWidget(self.ctx_slider, 0, 1)
        grid.addWidget(self.ctx_input, 0, 2)

        # GPU Layers
        grid.addWidget(QLabel("GPU Layers:"), 1, 0)
        self.gpu_slider = QSlider(Qt.Horizontal)
        self.gpu_slider.setRange(0, 100) # Updated dynamically
        self.gpu_slider.valueChanged.connect(self.update_gpu_label)
        self.gpu_slider.valueChanged.connect(self.update_vram_display)
        self.gpu_label = QLabel("99")
        self.gpu_label.setFixedWidth(50)
        grid.addWidget(self.gpu_slider, 1, 1)
        grid.addWidget(self.gpu_label, 1, 2)
        
        # Cache Type
        grid.addWidget(QLabel("KV Cache:"), 2, 0)
        self.cache_combo = QComboBox()
        self.cache_combo.addItems(["f16", "q8_0", "q4_0"])
        self.cache_combo.currentTextChanged.connect(self.update_vram_display)
        grid.addWidget(self.cache_combo, 2, 1)
        
        
        # Flash Attention
        self.flash_check = QCheckBox("Flash Attention")
        self.flash_check.toggled.connect(self.update_vram_display)
        grid.addWidget(self.flash_check, 3, 0, 1, 2)
        
        scroll_layout.addWidget(settings_group)

        # VRAM Bar
        self.vram_card = Card("VRAM Estimation")
        self.vram_bar = VramBar()
        self.vram_card.add_widget(self.vram_bar)
        scroll_layout.addWidget(self.vram_card)
        
        # Vision Model Settings (MCCC Refactor: Uses encapsulated widget)
        self.vision_settings_widget = VisionSettingsWidget()
        scroll_layout.addWidget(self.vision_settings_widget)
        self.vision_settings_widget.hide() # Hide by default
        
        # GPU Selection
        self.gpu_group = QGroupBox("Active GPUs")
        self.gpu_layout = QVBoxLayout(self.gpu_group) # Vertical list of checkboxes
        scroll_layout.addWidget(self.gpu_group)
        
        # Network & Security
        net_group = QGroupBox("Network & Security")
        net_layout = QGridLayout(net_group)
        net_layout.setVerticalSpacing(10)
        
        net_layout.addWidget(QLabel("Port (Optional):"), 0, 0)
        self.port_input = QLineEdit()
        self.port_input.setPlaceholderText("Auto")
        self.port_input.setFixedWidth(100)
        net_layout.addWidget(self.port_input, 0, 1)

        # External Access Checkbox
        self.host_check = QCheckBox("Allow External (0.0.0.0)")
        self.host_check.setToolTip("If checked, server listens on all interfaces (accessible from LAN).")
        net_layout.addWidget(self.host_check, 0, 2)
        
        net_layout.addWidget(QLabel("API Key (Optional):"), 1, 0)
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-...")
        # self.api_key_input.setEchoMode(QLineEdit.Password) # Unmasked per user request
        net_layout.addWidget(self.api_key_input, 1, 1)
        
        scroll_layout.addWidget(net_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Actions Footer
        btn_layout = QHBoxLayout()
        btn_layout.addStretch() # Push button to right or center? Let's make it full width but distinct
        self.btn_launch_server = QPushButton("Launch Server")
        self.btn_launch_server.setFixedHeight(45)
        self.btn_launch_server.setStyleSheet("""
            QPushButton { 
                background-color: #198754; 
                color: white; 
                font-weight: bold; 
                border-radius: 6px; 
                margin-top: 10px;
            }
            QPushButton:hover { background-color: #157347; }
        """)
        self.btn_launch_server.clicked.connect(self.launch_server)
        btn_layout.addWidget(self.btn_launch_server)
        btn_layout.addStretch() # Center it effectively if we want, or just leave it
        layout.addLayout(btn_layout)

    # --- Slots & Logic ---

    def update_ctx_input(self, val):
        self.ctx_input.setText(str(val))

    def on_ctx_input_changed(self):
        text = self.ctx_input.text().strip()
        if text.isdigit():
            val = int(text)
            # Update slider without triggering loop if possible, 
            # but slider signal only fires on value change. 
            # If value is same, no signal. 
            # If value is different, signal fires -> updates text. 
            # Use blockSignals to be safe or just let it sync.
            # actually logic: text edited -> slider update -> slider signal -> text update.
            # Text update overwrites what user typed? No, user finished typing.
            # But if user typed 8000, slider might be 8192 step? 
            # Slider is 512 step. 8000 might snap.
            # We want to keep the precise value in text box.
            
            # Simple approach: Block signals of slider while setting it
            self.ctx_slider.blockSignals(True)
            self.ctx_slider.setValue(val)
            self.ctx_slider.blockSignals(False)
            
            # Also trigger vram update since we bypassed slider signal
            self.update_vram_display()
    
    def update_gpu_label(self, val):
        self.gpu_label.setText("All" if val >= self.gpu_slider.maximum() else str(val))


    def on_projector_changed(self, text):
        self.current_projector = self.projector_combo.currentData()
        self.update_vram_display()
    
    def refresh_projector_list(self):
        """Reload projector files from disk without changing the model."""
        if not self.current_model:
            return
        
        model_path = os.path.join(self.cfg["model_path"], self.current_model)
        
        # Save current selection
        current_proj = self.current_projector
        
        # Block signals to avoid triggering updates during reload
        self.projector_combo.blockSignals(True)
        self.projector_combo.clear()
        
        # Reload candidates
        candidates = list_projector_candidates(model_path)
        
        if candidates:
            for c in candidates:
                self.projector_combo.addItem(os.path.basename(c), c)
            
            # Try to restore previous selection if it still exists
            if current_proj and current_proj in candidates:
                idx = self.projector_combo.findData(current_proj)
                if idx >= 0:
                    self.projector_combo.setCurrentIndex(idx)
                    self.current_projector = current_proj
                else:
                    # Fallback to first item
                    self.projector_combo.setCurrentIndex(0)
                    self.current_projector = candidates[0]
            else:
                # Try to find best match
                best_match = find_projector_file(model_path)
                if best_match:
                    idx = self.projector_combo.findData(best_match)
                    if idx >= 0:
                        self.projector_combo.setCurrentIndex(idx)
                        self.current_projector = best_match
                    else:
                        self.projector_combo.setCurrentIndex(0)
                        self.current_projector = candidates[0]
                else:
                    self.projector_combo.setCurrentIndex(0)
                    self.current_projector = candidates[0]
            
            self.projector_combo.setEnabled(True)
        else:
            self.projector_combo.addItem("No projector found")
            self.projector_combo.setEnabled(False)
            self.current_projector = None
        
        self.projector_combo.blockSignals(False)
        
        # Update VRAM display with new projector
        self.update_vram_display()
    
    def _on_paths_saved(self):
        """Reload model list and reset selection after bin/model paths are changed."""
        self.models = list_models(self.cfg)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(self.models)
        self.model_combo.setCurrentIndex(-1)
        self.model_combo.blockSignals(False)
        self.current_model = None
        self.model_info_lbl.setText("Select a model...")

    def refresh_model_list(self):
        """Reload model list from disk without losing current state."""
        # Save current selection
        current_model = self.current_model
        
        # Block signals to avoid triggering on_model_changed during reload
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        
        # Reload models from disk
        self.models = list_models(self.cfg)
        self.model_combo.addItems(self.models)
        
        # Try to restore previous selection if it still exists
        if current_model and current_model in self.models:
            idx = self.model_combo.findText(current_model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            else:
                # Model was removed, clear selection
                self.model_combo.setCurrentIndex(-1)
                self.current_model = None
        else:
            # No previous selection or model was removed
            self.model_combo.setCurrentIndex(-1)
            self.current_model = None
        
        self.model_combo.blockSignals(False)
        
        # If we still have a valid model selected, trigger update
        if self.current_model:
            self.on_model_changed(self.current_model)

    def on_model_changed(self, model_name):
        if not model_name: return
        self.current_model = model_name
        model_path = os.path.join(self.cfg["model_path"], model_name)
        
        # Load capabilities & Architecture
        self.caps = get_model_capabilities(self.cfg, model_name)
        self.arch_info = get_model_architecture(model_path)
        self.safe_defaults = get_safe_defaults(self.caps["family"])
        
        # Vision check
        self.current_projector = None
        self.projector_combo.blockSignals(True)
        self.projector_combo.clear()
        self.projector_combo.setEnabled(True)
        
        # Force vision if filename contains hints (User request: rely on filename)
        name_lower = model_name.lower()
        forced_vision = False
        if any(x in name_lower for x in ["vl", "llava", "minicpm", "vision", "moondream", "bunnies"]):
            forced_vision = True
        
        if self.arch_info.get("has_vision", False) or forced_vision:
            self.projector_container.show()
            candidates = list_projector_candidates(model_path)
            
            if candidates:
                for c in candidates:
                    self.projector_combo.addItem(os.path.basename(c), c)
                
                # Try to find best match automatically
                best_match = find_projector_file(model_path)
                if best_match:
                    idx = self.projector_combo.findData(best_match)
                    if idx >= 0:
                        self.projector_combo.setCurrentIndex(idx)
                        self.current_projector = best_match
                    else:
                        self.projector_combo.setCurrentIndex(0)
                        self.current_projector = candidates[0]
                else:
                    self.projector_combo.setCurrentIndex(0)
                    self.current_projector = candidates[0]
            else:
                 self.projector_combo.addItem("No projector found")
                 self.projector_combo.setEnabled(False)
        else:
            self.projector_container.hide()
        
        # Show/hide vision settings using widget logic
        is_vision = self.arch_info.get("has_vision", False) or forced_vision
        self.vision_settings_widget.update_visibility(model_name, is_vision)
        
        self.projector_combo.blockSignals(False)
        
        # Update UI Info
        fam = self.caps['family']
        layers = self.arch_info.get('n_layers', '?')
        params = self.arch_info.get('parameter_count', '?')
        self.model_info_lbl.setText(f"Family: {fam} | Layers: {layers} | Architecture: {self.arch_info.get('architecture', 'unknown')}")

        total_layers = self.arch_info.get('n_layers', 100)
        saved = self.cfg.get("model_defaults", {}).get(model_name)

        if saved:
            self.ctx_slider.setValue(int(saved.get("ctx_size", self.safe_defaults['ctx_size'])))
            self.gpu_slider.setValue(int(saved.get("gpu_layers", total_layers)))

            flash_val = saved.get("flash_attention", "off")
            is_flash_on = (flash_val == "on" or flash_val is True)
            self.flash_check.setChecked(is_flash_on if self.caps['flash_ok'] else False)

            self.cache_combo.setCurrentText(saved.get("cache_type", self.safe_defaults['cache_type']))

            self.port_input.setText(str(saved.get("port", "")))
            self.api_key_input.setText(str(saved.get("api_key", "")))

            h_val = saved.get("host", True)
            if isinstance(h_val, str):
                h_val = (h_val.lower() == 'true')
            self.host_check.setChecked(bool(h_val))

            self.vision_settings_widget.load_defaults(saved, model_name)
            selected_gpus = saved.get("selected_gpus")
        else:
            self.ctx_slider.setValue(self.safe_defaults['ctx_size'])
            self.gpu_slider.setValue(total_layers)

            self.flash_check.setEnabled(self.caps['flash_ok'])
            self.flash_check.setChecked(
                self.caps['flash_ok'] and self.safe_defaults['flash_attention'] != 'off'
            )

            self.cache_combo.setCurrentText(self.safe_defaults['cache_type'])
            self.port_input.clear()
            self.api_key_input.clear()
            self.host_check.setChecked(True)
            selected_gpus = None

        self.refresh_gpu_list(selected_gpus)
        self.update_vram_display()

    def refresh_gpu_list(self, selected_indices=None):
        # Clear existing layout items
        while self.gpu_layout.count():
            item = self.gpu_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            
        self.gpu_checks = []
        # Get detected GPUs from capabilities (detectors.py)
        gpus = self.caps.get('system_flags', {}).get('gpus', [])
        
        if not gpus:
             lbl = QLabel("No detailed GPU info available (using basic detection)")
             lbl.setStyleSheet("color: #666; font-style: italic;")
             self.gpu_layout.addWidget(lbl)
             return 

        for gpu in gpus:
            name = gpu.get('name', f"GPU {gpu['index']}")
            vram = gpu.get('vram_gb', 0)
            cb = QCheckBox(f"GPU {gpu['index']}: {name} ({vram:.2f} GB)")
            
            # Determine checked state
            if selected_indices is not None:
                cb.setChecked(gpu['index'] in selected_indices)
            else:
                cb.setChecked(True) # Default to all enabled if no pref
            
            cb.stateChanged.connect(self.update_vram_display)
            self.gpu_layout.addWidget(cb)
            # Store reference to GPU data and the checkbox
            self.gpu_checks.append((gpu, cb))

    def update_vram_display(self):
        if not self.current_model: return
        
        try:
            # Gather settings
            # Gather settings
            try:
                ctx_val = int(self.ctx_input.text())
            except:
                ctx_val = self.ctx_slider.value()

            settings = {
                "ctx_size": ctx_val,
                "gpu_layers": self.gpu_slider.value(),
                "flash_attention": self.flash_check.isChecked(),
                "cache_type": self.cache_combo.currentText(),
                "projector_path": self.current_projector
            }
            
            # Merge vision settings
            v_settings = self.vision_settings_widget.get_settings()
            if v_settings:
                settings.update(v_settings)
            
            # Calc usage
            usage = calculate_vram_usage(
                os.path.join(self.cfg["model_path"], self.current_model),
                settings,
                self.arch_info
            )
            
            # Calc total available from selected GPUs
            total_vram = 0.0
            if self.gpu_checks:
                for gpu, cb in self.gpu_checks:
                    if cb.isChecked():
                        total_vram += gpu['vram_gb']
                if total_vram == 0:
                     # If all unchecked, 0
                     pass
            else:
                # Fallback for single GPU systems / no detailed detection
                total_vram = self.caps.get('vram_gb', 0)
                
            available = get_vram_available(total_vram)
            self.vram_bar.update_usage(usage['total'], available)
            
            # status update (Check if lbl exists first)
            if hasattr(self, 'status_lbl'):
                if usage['total'] > available:
                    diff = usage['total'] - available
                    # Shortened to fit: "⚠️ Limit: 12.00/12.00 GB (+0.45 GB)"
                    self.status_lbl.setText(f"⚠️ Limit: {usage['total']:.2f}/{total_vram:.2f} GB (+{diff:.2f} GB)")
                    self.status_lbl.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 12px;")
                else:
                    self.status_lbl.setText(f"Est. VRAM: {usage['total']:.2f} / {total_vram:.2f} GB Used ({available - usage['total']:.2f} GB Free)")
                    self.status_lbl.setStyleSheet("color: #198754; font-size: 12px;")
        except Exception as e:
            print(f"Error updating VRAM display: {e}")

    def launch_server(self):
        # Determine Port
        user_port = self.port_input.text().strip()
        if user_port and user_port.isdigit():
            port = int(user_port)
        else:
            # Find next free port
            port = self.server_manager.find_free_port(self.next_port)
            self.next_port = port + 1
            
        # API Key
        api_key = self.api_key_input.text().strip()
        if not api_key: api_key = None
        
        # Get context size from input for precision
        try:
            ctx_size = int(self.ctx_input.text())
        except:
            ctx_size = self.ctx_slider.value()

        # Build command
        host_arg = "0.0.0.0" if self.host_check.isChecked() else "127.0.0.1"
        
        # Prepare vision settings if this is a vision model
        # Prepare vision settings if this is a vision model
        vision_config = None
        if not self.vision_settings_widget.isHidden():
             vision_data = self.vision_settings_widget.get_settings()
             if vision_data:
                 vision_config = VisionConfig.from_dict(vision_data)

        builder = ServerCommandBuilder(self.cfg)
        cmd = builder.build(
            model_file=self.current_model,
            ctx_size=ctx_size,
            gpu_layers=str(self.gpu_slider.value()),
            flash=self.flash_check.isChecked(),
            cache=self.cache_combo.currentText() != "f16",
            cache_type=self.cache_combo.currentText(),
            extra_args=f"--port {port}", 
            projector_file=self.current_projector,
            api_key=api_key,
            host=host_arg,
            vision_config=vision_config
        )
        
        # Build Environment
        env = QProcess.systemEnvironment() # Get base system env
        selected_indices = []
        
        if self.gpu_checks:
            for gpu, cb in self.gpu_checks:
                if cb.isChecked():
                    selected_indices.append(str(gpu['index']))
            
            # If user selected specific GPUs, set CUDA_VISIBLE_DEVICES
            if selected_indices:
                env_list = env 
                # Remove existing CUDA_VISIBLE_DEVICES to be safe
                env_list = [e for e in env_list if not e.startswith("CUDA_VISIBLE_DEVICES=")]
                # Force PCI_BUS_ID ordering to match nvidia-smi (detectors.py)
                env_list = [e for e in env_list if not e.startswith("CUDA_DEVICE_ORDER=")]
                env_list.append("CUDA_DEVICE_ORDER=PCI_BUS_ID")
                env_list.append(f"CUDA_VISIBLE_DEVICES={','.join(selected_indices)}")
                env = env_list

        # Save Settings to Config (Sticky Settings)
        if "model_defaults" not in self.cfg:
            self.cfg["model_defaults"] = {}
            
        self.cfg["model_defaults"][self.current_model] = {
            "ctx_size": ctx_size,
            "gpu_layers": self.gpu_slider.value(),
            "flash_attention": "on" if self.flash_check.isChecked() else "off",
            "cache_type": self.cache_combo.currentText(),
            "port": self.port_input.text().strip(),
            "api_key": self.api_key_input.text().strip(),
            "host": self.host_check.isChecked(),
            "selected_gpus": [int(x) for x in selected_indices]
        }
        
        # Save Vision Settings (Mix-in)
        v_settings = self.vision_settings_widget.get_settings()
        if v_settings:
            # Prefix keys with vision_ for compatibility
            for k, v in v_settings.items():
                self.cfg["model_defaults"][self.current_model][f"vision_{k}"] = v
        
        save_config(self.cfg)

        # Calculate Offload %
        total_layers = self.arch_info.get('n_layers', 0)
        user_layers = self.gpu_slider.value()
        
        offload_str = "N/A"
        if total_layers > 0:
            if user_layers >= total_layers:
                offload_str = "100% (All Layers)"
            elif user_layers == 0:
                offload_str = "0% (CPU)"
            else:
                pct = int((user_layers / total_layers) * 100)
                offload_str = f"{pct}% ({user_layers}/{total_layers})"

        # Start via Manager
        # Start via Manager (Signal/Slot preferred for thread safety, but direct call works if method is thread-safe or simple)
        # Since ServerManager is in another thread, we should invokeMethod or use a signal. 
        # But for now, PySide handles cross-thread calls by queuing them if arguments are metatypes?
        # Ideally, we emit a signal 'request_launch' and MANAGER connects to it.
        # Calling methods on object in another thread directly RUNS IN CALLER THREAD unless using invokeMethod.
        # We need to use Signals!
        
        # Quick Refactor: Define signal in MainWindow
        # self.emit_launch_server(...)
        # For this step, I'll use QMetaObject.invokeMethod to be safe, or just let it run.
        # Actually, if we call it directly, it runs in MainThread. 
        # ServerSession created in MainThread will live in MainThread.
        # We want Session to live in ManagerThread.
        # So we MUST use signal to trigger creation in ManagerThread.
        
        # Let's fix this properly.
        # I will add `request_launch` signal to MainWindow and connect it.
        try:
            self.request_launch.emit(
                port,
                self.current_model,
                ctx_size,
                str(self.gpu_slider.value()),
                cmd,
                env,
                {"offload_confirmed": offload_str}
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch request: {e}")
            return
        
        # Switch to logs (Index 5)
        self.content_stack.setCurrentWidget(self.page_logs)
        self.switch_tab(5, self.nav_btns[5])
        
        self.log_output.append(f"\n> [System] Launching {self.current_model} on Port {port}...")
        if selected_indices:
             self.log_output.append(f"> [System] GPU(s): {','.join(selected_indices)}")

            
    def find_free_port(self, start_port):
        return self.server_manager.find_free_port(start_port)

    def handle_session_log(self, port, msg):
        # Prepend port to message
        formatted = f"[{port}] {msg}"
        self.log_output.moveCursor(QTextCursor.End)
        self.log_output.insertPlainText(formatted)
        self.log_output.ensureCursorVisible()

    def handle_session_started(self, info):
        self.page_home.update_session_status(True, info)
        self.log_output.append(f"\n> [System] Server started on port {info['port']}")

    def handle_session_stats(self, port, stats):
        self.page_home.update_session_stats(port, stats)

    def handle_session_stopped(self, port):
        # Update Home Page
        self.page_home.update_session_status(False, {"port": port})
        self.log_output.append(f"\n> [System] Server on port {port} stopped.")
        self.update_vram_display()

    # handle_session_status Removed (replaced by started/stopped signals)

    def refresh_home_sessions(self):
        self.page_home.clear_sessions()
        sessions = self.server_manager.get_active_sessions()
        if not sessions:
            self.page_home.render_empty_state()
            return
            
        for info in sessions:
            self.page_home.render_active_session(info)

    def stop_server_by_port(self, port):
        self.request_stop_server.emit(str(port))

    def stop_all_servers(self):
        self.request_stop_all.emit()

    def switch_to_logs(self, port):
        # Later we could filter logs by port
        self.content_stack.setCurrentWidget(self.page_logs)
        self.switch_tab(5, self.nav_btns[5])

    def closeEvent(self, event):
        if self.manager_thread.isRunning():
            self.manager_thread.quit()
            self.manager_thread.wait(2000)
        event.accept()

