from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLabel, 
                               QSlider, QLineEdit, QCheckBox, QGroupBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator

class VisionSettingsWidget(QWidget):
    """
    Dedicated widget for managing Vision Language Model (VLM) settings.
    MCCC Compliance: Encapsulates all vision-related UI logic and state.
    """
    
    # Signals
    settings_changed = Signal()      # Emitted when any setting changes
    vram_changed = Signal()          # Emitted when a setting affecting VRAM changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.group = QGroupBox("Vision Model Settings")
        grid = QGridLayout(self.group)
        grid.setVerticalSpacing(15)
        grid.setHorizontalSpacing(20)

        # Architecture hint label
        self.vision_hint_lbl = QLabel("")
        self.vision_hint_lbl.setStyleSheet("color: #ffc107; font-style: italic; font-size: 11px;")
        self.vision_hint_lbl.setWordWrap(True)
        grid.addWidget(self.vision_hint_lbl, 0, 0, 1, 3)

        # Image Resolution
        grid.addWidget(QLabel("Image Resolution:"), 1, 0)
        self.img_res_slider = QSlider(Qt.Horizontal)
        self.img_res_slider.setRange(256, 2048)
        self.img_res_slider.setSingleStep(128)
        self.img_res_slider.setValue(1024)
        self.img_res_slider.valueChanged.connect(self._sync_res_input)
        self.img_res_slider.valueChanged.connect(lambda _: self.vram_changed.emit())

        self.img_res_input = QLineEdit("1024")
        self.img_res_input.setFixedWidth(60)
        self.img_res_input.setAlignment(Qt.AlignCenter)
        self.img_res_input.setValidator(QIntValidator(256, 2048, self))
        self.img_res_input.editingFinished.connect(self._sync_res_slider)

        grid.addWidget(self.img_res_slider, 1, 1)
        grid.addWidget(self.img_res_input, 1, 2)

        # Batch Size
        grid.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_slider = QSlider(Qt.Horizontal)
        self.batch_size_slider.setRange(256, 2048)
        self.batch_size_slider.setSingleStep(128)
        self.batch_size_slider.setValue(1024)
        self.batch_size_slider.valueChanged.connect(self._sync_batch_input)
        self.batch_size_slider.valueChanged.connect(lambda _: self.vram_changed.emit())

        self.batch_size_input = QLineEdit("1024")
        self.batch_size_input.setFixedWidth(60)
        self.batch_size_input.setAlignment(Qt.AlignCenter)
        self.batch_size_input.setValidator(QIntValidator(256, 2048, self))
        self.batch_size_input.editingFinished.connect(self._sync_batch_slider)

        grid.addWidget(self.batch_size_slider, 2, 1)
        grid.addWidget(self.batch_size_input, 2, 2)

        # Micro-batch Size
        grid.addWidget(QLabel("Micro-batch Size:"), 3, 0)
        self.ubatch_size_slider = QSlider(Qt.Horizontal)
        self.ubatch_size_slider.setRange(128, 1024)
        self.ubatch_size_slider.setSingleStep(64)
        self.ubatch_size_slider.setValue(512)
        self.ubatch_size_slider.valueChanged.connect(self._sync_ubatch_input)
        self.ubatch_size_slider.valueChanged.connect(lambda _: self.vram_changed.emit())

        self.ubatch_size_input = QLineEdit("512")
        self.ubatch_size_input.setFixedWidth(60)
        self.ubatch_size_input.setAlignment(Qt.AlignCenter)
        self.ubatch_size_input.setValidator(QIntValidator(128, 1024, self))
        self.ubatch_size_input.editingFinished.connect(self._sync_ubatch_slider)

        grid.addWidget(self.ubatch_size_slider, 3, 1)
        grid.addWidget(self.ubatch_size_input, 3, 2)

        # MiniCPM Specifics — all three checkboxes together, same level
        self.minicpm_container = QWidget()
        mc_layout = QVBoxLayout(self.minicpm_container)
        mc_layout.setContentsMargins(0, 5, 0, 0)
        mc_layout.setSpacing(8)

        self.no_mmproj_offload_check = QCheckBox("Keep Projector on GPU")
        self.no_mmproj_offload_check.setToolTip("Offloads the vision projector to GPU. Uncheck to keep on CPU (slow).")
        self.no_mmproj_offload_check.toggled.connect(lambda _: self.vram_changed.emit())
        mc_layout.addWidget(self.no_mmproj_offload_check)

        self.disable_thinking_check = QCheckBox("Disable Thinking Mode (--reasoning-budget 0)")
        self.disable_thinking_check.setToolTip("Sets reasoning budget to 0 to disable automatic thinking loop.")
        mc_layout.addWidget(self.disable_thinking_check)

        self.force_jinja_check = QCheckBox("Force Jinja Template (Meta Override)")
        self.force_jinja_check.setToolTip("Overrides GGUF metadata. Forces Jinja + disables reasoning format. Fixes OCR.")
        mc_layout.addWidget(self.force_jinja_check)

        grid.addWidget(self.minicpm_container, 4, 0, 1, 3)
        self.minicpm_container.hide()

        layout.addWidget(self.group)

    # --- Sync Logic ---
    def _sync_res_input(self, val): self.img_res_input.setText(str(val))
    def _sync_res_slider(self): self._sync_slider(self.img_res_input, self.img_res_slider)
    
    def _sync_batch_input(self, val): self.batch_size_input.setText(str(val))
    def _sync_batch_slider(self): self._sync_slider(self.batch_size_input, self.batch_size_slider)
    
    def _sync_ubatch_input(self, val): self.ubatch_size_input.setText(str(val))
    def _sync_ubatch_slider(self): self._sync_slider(self.ubatch_size_input, self.ubatch_size_slider)

    def _sync_slider(self, input_widget, slider_widget):
        try:
            val = int(input_widget.text())
            slider_widget.setValue(val)
        except ValueError:
            input_widget.setText(str(slider_widget.value()))

    # --- Public API ---

    def update_visibility(self, model_name: str, has_vision: bool):
        """
        Updates visibility of the entire group and sub-sections based on model.
        """
        if not has_vision:
            self.hide()
            return
            
        self.show()
        name_lower = model_name.lower()
        
        # MiniCPM Logic
        if 'minicpm' in name_lower:
            self.vision_hint_lbl.setText("⚠ CRITICAL: Tiled high-res reading. Wrong resolution = massive failure.")
            self.vision_hint_lbl.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 11px;")
            self.minicpm_container.show()
        elif any(x in name_lower for x in ['qwen', 'pixtral', 'llava']):
            self.vision_hint_lbl.setText("Note: Auto-resizes internally. Resolution setting has minimal effect.")
            self.vision_hint_lbl.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
            self.minicpm_container.hide()
        else:
            self.vision_hint_lbl.setText("Model load parameters for vision encoder")
            self.vision_hint_lbl.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
            self.minicpm_container.hide()

    def get_settings(self) -> dict:
        """Returns MCCC-compliant settings dictionary"""
        if self.isHidden():
            return None
            
        return {
            "image_resolution": self.img_res_slider.value(),
            "batch_size": self.batch_size_slider.value(),
            "ubatch_size": self.ubatch_size_slider.value(),
            "no_mmproj_offload": not self.no_mmproj_offload_check.isChecked(),
            "disable_thinking": self.disable_thinking_check.isChecked() if self.minicpm_container.isVisible() else False,
            "force_jinja": self.force_jinja_check.isChecked() if self.minicpm_container.isVisible() else False
        }

    def load_defaults(self, defaults: dict, model_name: str):
        """Loads saved defaults safely"""
        if not defaults:
            # Set hard defaults
            self.img_res_slider.setValue(1024)
            self.batch_size_slider.setValue(1024)
            self.ubatch_size_slider.setValue(512)
            self.no_mmproj_offload_check.setChecked(True)   # default: GPU
            self.disable_thinking_check.setChecked(False)
            self.force_jinja_check.setChecked(False)
            return

        self.img_res_slider.setValue(int(defaults.get("vision_image_resolution", 1024)))
        self.batch_size_slider.setValue(int(defaults.get("vision_batch_size", 1024)))
        self.ubatch_size_slider.setValue(int(defaults.get("vision_ubatch_size", 512)))
        self.no_mmproj_offload_check.setChecked(not defaults.get("vision_no_mmproj_offload", False))
        
        # Robust MiniCPM restore
        if 'minicpm' in model_name.lower():
            self.disable_thinking_check.setChecked(defaults.get("vision_disable_thinking", False))
            self.force_jinja_check.setChecked(defaults.get("vision_force_jinja", False))
        else:
            self.disable_thinking_check.setChecked(False)
            self.force_jinja_check.setChecked(False)
