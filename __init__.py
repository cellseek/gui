"""
CellSeek GUI - A modern PyQt6 interface for frame-by-frame cell segmentation and tracking

This module provides the main application class and entry point for the
CellSeek GUI application that integrates CellSAM, SAM, and CUTIE for frame-by-frame processing.
"""

import sys
from pathlib import Path

# Add project paths to sys.path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sam_path = project_root / "sam"
cutie_path = project_root / "cutie"

sys.path.insert(0, str(sam_path.resolve()))
sys.path.insert(0, str(cutie_path.resolve()))

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from new_main_window import NewMainWindow

__version__ = "2.0.0"
__author__ = "CellSeek Team"


class CellSeekApp(QApplication):
    """Frame-by-frame application class for CellSeek GUI"""

    def __init__(self, argv):
        super().__init__(argv)

        # Set application properties
        self.setApplicationName("CellSeek Frame-by-Frame")
        self.setApplicationVersion(__version__)
        self.setOrganizationName("CellSeek Team")

        # Set application icon if available
        icon_path = current_dir / "resources" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Apply dark theme
        self.setStyleSheet(self._get_dark_theme())

        # Create main window
        self.main_window = NewMainWindow()
        self.main_window.show()

    def _get_dark_theme(self):
        """Return dark theme stylesheet"""
        return """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 9pt;
        }
        
        QStackedWidget {
            background-color: #2b2b2b;
        }
        
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #106ebe;
        }
        
        QPushButton:pressed {
            background-color: #005a9e;
        }
        
        QPushButton:disabled {
            background-color: #404040;
            color: #808080;
        }
        
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #404040;
            border: 1px solid #606060;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #0078d4;
        }
        
        QProgressBar {
            background-color: #404040;
            border: 1px solid #606060;
            border-radius: 4px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 3px;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #606060;
            border-radius: 4px;
            margin-top: 1ex;
            padding-top: 8px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QListWidget {
            background-color: #353535;
            border: 1px solid #606060;
            border-radius: 4px;
            alternate-background-color: #404040;
        }
        
        QListWidget::item {
            padding: 4px;
            border-bottom: 1px solid #505050;
        }
        
        QListWidget::item:selected {
            background-color: #0078d4;
        }
        
        QListWidget::item:hover {
            background-color: #454545;
        }
        
        QScrollBar:vertical {
            background: #404040;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background: #606060;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #707070;
        }
        
        QStatusBar {
            background-color: #353535;
            border-top: 1px solid #606060;
        }
        
        QLabel {
            background: transparent;
            border: none;
        }
        
        QRadioButton {
            spacing: 8px;
        }
        
        QRadioButton::indicator {
            width: 13px;
            height: 13px;
        }
        
        QRadioButton::indicator:unchecked {
            border: 2px solid #606060;
            border-radius: 7px;
            background-color: #404040;
        }
        
        QRadioButton::indicator:checked {
            border: 2px solid #0078d4;
            border-radius: 7px;
            background-color: #0078d4;
        }
        
        QCheckBox::indicator {
            width: 13px;
            height: 13px;
        }
        
        QCheckBox::indicator:unchecked {
            border: 2px solid #606060;
            border-radius: 2px;
            background-color: #404040;
        }
        
        QCheckBox::indicator:checked {
            border: 2px solid #0078d4;
            border-radius: 2px;
            background-color: #0078d4;
        }
        """


def main():
    """Entry point for the CellSeek GUI application"""
    app = CellSeekApp(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
