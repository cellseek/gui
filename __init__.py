"""
CellSeek GUI - A modern PyQt6 interface for cell segmentation and tracking

This module provides the main application class and entry point for the
CellSeek GUI application that integrates CellSAM, XMem, and analysis tools.
"""

import sys
from pathlib import Path

# Add project paths to sys.path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sam_path = project_root / "sam"
xmem_path = project_root / "xmem"
seek_path = project_root / "seek"

sys.path.insert(0, str(sam_path.resolve()))
sys.path.insert(0, str(xmem_path.resolve()))
sys.path.insert(0, str(seek_path.resolve()))

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from main_window import MainWindow

__version__ = "1.0.0"
__author__ = "CellSeek Team"


class CellSeekApp(QApplication):
    """Main application class for CellSeek GUI"""

    def __init__(self, argv):
        super().__init__(argv)

        # Set application properties
        self.setApplicationName("CellSeek")
        self.setApplicationVersion(__version__)
        self.setOrganizationName("CellSeek Team")

        # Set application icon if available
        icon_path = current_dir / "resources" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Apply dark theme
        self.setStyleSheet(self._get_dark_theme())

        # Create main window
        self.main_window = MainWindow()
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
        
        QTabWidget::pane {
            border: 1px solid #3d3d3d;
            background-color: #353535;
        }
        
        QTabBar::tab {
            background-color: #404040;
            color: #ffffff;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background-color: #0078d4;
        }
        
        QTabBar::tab:hover {
            background-color: #505050;
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
        
        QListWidget, QTreeWidget, QTableWidget {
            background-color: #353535;
            border: 1px solid #606060;
            border-radius: 4px;
            alternate-background-color: #404040;
        }
        
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
            background-color: #0078d4;
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
        
        QMenuBar {
            background-color: #353535;
            border-bottom: 1px solid #606060;
        }
        
        QMenuBar::item {
            padding: 4px 8px;
        }
        
        QMenuBar::item:selected {
            background-color: #0078d4;
        }
        
        QMenu {
            background-color: #353535;
            border: 1px solid #606060;
        }
        
        QMenu::item {
            padding: 4px 16px;
        }
        
        QMenu::item:selected {
            background-color: #0078d4;
        }
        
        QStatusBar {
            background-color: #353535;
            border-top: 1px solid #606060;
        }
        """


def main():
    """Entry point for the CellSeek GUI application"""
    app = CellSeekApp(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
