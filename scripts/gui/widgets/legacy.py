from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QProgressBar, QFrame)
from PySide6.QtCore import Qt

class VramBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Labels
        self.limit_label = QLabel("VRAM Usage: 0.00 / 0.00 GB")
        layout.addWidget(self.limit_label)
        
        # Bar
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)
        
        # Legend (simplified)
        legend = QHBoxLayout()
        legend.addWidget(QLabel("Model"))
        layout.addLayout(legend)
        
    def update_usage(self, used_gb, total_gb, details=None):
        if total_gb > 0:
            percent = int((used_gb / total_gb) * 100)
        else:
            percent = 0
            
        self.bar.setValue(min(percent, 100))
        self.limit_label.setText(f"VRAM Usage: {used_gb:.2f} / {total_gb:.2f} GB")
        
        # Color coding
        if used_gb > total_gb:
            self.bar.setStyleSheet("QProgressBar::chunk { background-color: #dc3545; }") # Red
        else:
            self.bar.setStyleSheet("QProgressBar::chunk { background-color: #198754; }") # Green

class ResourceMeter(QWidget):
    """
    Double bar meter for System RAM and VRAM.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(15)
        
        # -- System RAM --
        ram_layout = QVBoxLayout()
        ram_layout.setSpacing(2)
        
        self.ram_lbl = QLabel("System RAM (Process): Waiting...")
        self.ram_lbl.setToolTip("Actual System RAM used by the llama-server process (not VRAM).")
        self.ram_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        ram_layout.addWidget(self.ram_lbl)
        
        self.ram_bar = QProgressBar()
        # ... existing bar setup ...
        self.ram_bar.setRange(0, 64) 
        self.ram_bar.setTextVisible(False)
        self.ram_bar.setFixedHeight(8)
        self.ram_bar.setStyleSheet("QProgressBar::chunk { background-color: #0d6efd; } QProgressBar { background: #333; border: none; }")
        ram_layout.addWidget(self.ram_bar)
        
        layout.addLayout(ram_layout, 1)

        # -- Offload Status --
        offload_layout = QVBoxLayout()
        offload_layout.setSpacing(2)
        
        self.offload_lbl = QLabel("GPU Offload: Pending...")
        self.offload_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        self.offload_lbl.setAlignment(Qt.AlignRight)
        offload_layout.addWidget(self.offload_lbl)
        
        # We could add a mini bar for offload completion?
        # For now just text is fine as requested.
        
        layout.addLayout(offload_layout, 0) # 0 stretch to keep it tight
        
    def update_ram(self, gb):
        self.ram_lbl.setText(f"System RAM (Process): {gb:.2f} GB")
        self.ram_lbl.setStyleSheet("color: #0d6efd; font-size: 11px; font-weight: bold;")
        self.ram_bar.setValue(int(gb))

    def update_offload(self, status: str):
        # status format: "33/33"
        try:
            done, total = map(int, status.split('/'))
            self.offload_lbl.setText(f"GPU: {done}/{total} Layers")
            
            if done == total and total > 0:
                self.offload_lbl.setStyleSheet("color: #198754; font-weight: bold; font-size: 11px;") # Green
            elif done == 0:
                self.offload_lbl.setStyleSheet("color: #dc3545; font-weight: bold; font-size: 11px;") # Red
            else:
                self.offload_lbl.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 11px;") # Orange
        except:
             self.offload_lbl.setText(f"GPU: {status}")

class Card(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(".Card { background-color: #252525; border-radius: 8px; border: 1px solid #3d3d3d; }")
        
        self.layout = QVBoxLayout(self)
        
        if title:
            title_lbl = QLabel(title)
            title_lbl.setStyleSheet("font-weight: bold; color: #0d6efd; font-size: 16px;")
            self.layout.addWidget(title_lbl)
            
    def add_widget(self, widget):
        self.layout.addWidget(widget)
    
    def add_layout(self, layout):
        self.layout.addLayout(layout)
