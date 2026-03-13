#!/usr/bin/env python3
import sys
import os
import signal
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from scripts.gui.main_window import MainWindow
from scripts.gui.styles import get_stylesheet

def main():
    # Allow Ctrl+C to kill the app
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    app.setApplicationName("Llama Manager")
    app.setStyle("Fusion")
    
    # Apply custom dark theme
    app.setStyleSheet(get_stylesheet())
    
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
