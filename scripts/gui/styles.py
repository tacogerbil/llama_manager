def get_stylesheet():
    return """
    QMainWindow {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    QWidget {
        background-color: #1e1e1e;
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
    }
    QPushButton {
        background-color: #0d6efd;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #0b5ed7;
    }
    QPushButton:pressed {
        background-color: #0a58ca;
    }
    QPushButton#SecondaryButton {
        background-color: #6c757d;
    }
    QPushButton#SecondaryButton:hover {
        background-color: #5c636a;
    }
    QLabel {
        color: #f0f0f0;
    }
    QLineEdit {
        background-color: #2d2d2d;
        border: 1px solid #4d4d4d;
        border-radius: 4px;
        padding: 4px;
        color: #f0f0f0;
    }
    QComboBox {
        background-color: #2d2d2d;
        border: 1px solid #4d4d4d;
        border-radius: 4px;
        padding: 4px;
        color: #f0f0f0;
    }
    QComboBox::drop-down {
        border: none;
    }
    QSlider::groove:horizontal {
        border: 1px solid #3d3d3d;
        height: 8px;
        background: #2d2d2d;
        margin: 2px 0;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #0d6efd;
        border: 1px solid #0d6efd;
        width: 18px;
        height: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }
    QProgressBar {
        border: 1px solid #4d4d4d;
        border-radius: 4px;
        text-align: center;
        background-color: #2d2d2d;
    }
    QProgressBar::chunk {
        background-color: #0d6efd;
        border-radius: 3px;
    }
    QGroupBox {
        border: 1px solid #4d4d4d;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 10px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: #0d6efd;
    }
    QCheckBox {
        color: #f0f0f0;
        spacing: 5px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #4d4d4d;
        border-radius: 3px;
        background-color: #2d2d2d;
    }
    QCheckBox::indicator:checked {
        background-color: #0d6efd;
        border-color: #0d6efd;
    }
    QToolTip {
        background-color: #2d2d2d;
        color: #f0f0f0;
        border: 1px solid #4d4d4d;
        border-radius: 4px;
        padding: 4px 8px;
    }
    """
