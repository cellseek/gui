"""
Main window for the CellSeek GUI application
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from core.project_manager import ProjectManager
from widgets.analysis_panel import AnalysisPanel
from widgets.export_panel import ExportPanel
from widgets.frame_manager import FrameManagerWidget
from widgets.segmentation_panel import SegmentationPanel
from widgets.tracking_panel import TrackingPanel


class ModelLoadWorker(QThread):
    """Worker thread for loading models in the background"""

    progress_update = pyqtSignal(str)  # status message
    model_loaded = pyqtSignal(str, object)  # model_name, model_object
    all_models_loaded = pyqtSignal(dict)  # all_models_dict
    error_occurred = pyqtSignal(str, str)  # model_name, error_message

    def __init__(self):
        super().__init__()
        self._models = {}

    def run(self):
        """Load all models in background thread"""
        try:
            self.progress_update.emit("Loading models in background...")

            # Load CellSAM
            try:
                self.progress_update.emit("Loading CellSAM...")
                from sam import CellSAM

                device = "cuda" if torch.cuda.is_available() else "cpu"
                cellsam = CellSAM(device=device)
                self._models["cellsam"] = cellsam
                self.model_loaded.emit("cellsam", cellsam)
                self.progress_update.emit("CellSAM loaded successfully")
            except Exception as e:
                error_msg = f"Could not load CellSAM: {str(e)}"
                print(f"Warning: {error_msg}")
                self.error_occurred.emit("cellsam", error_msg)

            # Load SAM
            try:
                self.progress_update.emit("Loading SAM...")
                from segment_anything import SamPredictor, sam_model_registry

                # Look for SAM checkpoint
                possible_paths = [
                    "checkpoints/sam_vit_h_4b8939.pth",
                    "checkpoints/sam_vit_l_0b3195.pth",
                    "checkpoints/sam_vit_b_01ec64.pth",
                ]

                sam_checkpoint = None
                model_type = "vit_h"

                for path in possible_paths:
                    if Path(path).exists():
                        sam_checkpoint = path
                        if "vit_l" in path:
                            model_type = "vit_l"
                        elif "vit_b" in path:
                            model_type = "vit_b"
                        elif "vit_h" in path:
                            model_type = "vit_h"
                        break

                if sam_checkpoint:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                    sam.to(device=device)
                    sam_predictor = SamPredictor(sam)
                    self._models["sam"] = sam_predictor
                    self._models["sam_model_type"] = model_type
                    self.model_loaded.emit("sam", sam_predictor)
                    self.progress_update.emit("SAM loaded successfully")
                else:
                    error_msg = "SAM checkpoint not found"
                    self.error_occurred.emit("sam", error_msg)

            except Exception as e:
                error_msg = f"Could not load SAM: {str(e)}"
                print(f"Warning: {error_msg}")
                self.error_occurred.emit("sam", error_msg)

            # Load XMem
            try:
                self.progress_update.emit("Loading XMem...")
                from xmem import XMem

                # Setup checkpoint paths
                xmem_path = Path(__file__).parent.parent / "xmem"
                xmem_checkpoint = xmem_path / "checkpoints" / "XMem-s012.pth"

                # Create minimal args object
                class TrackArgs:
                    def __init__(self):
                        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                        self.sam_model_type = "vit_h"
                        self.debug = False
                        self.mask_save = False

                track_args = TrackArgs()

                if xmem_checkpoint.exists():
                    xmem = XMem(
                        xmem_checkpoint=str(xmem_checkpoint),
                        args=track_args,
                    )
                    self._models["xmem"] = xmem
                    self.model_loaded.emit("xmem", xmem)
                    self.progress_update.emit("XMem loaded successfully")
                else:
                    error_msg = "XMem checkpoint not found"
                    self.error_occurred.emit("xmem", error_msg)

            except Exception as e:
                error_msg = f"Could not load XMem: {str(e)}"
                print(f"Warning: {error_msg}")
                self.error_occurred.emit("xmem", error_msg)

            # Emit completion signal
            self.progress_update.emit("All models loaded")
            self.all_models_loaded.emit(self._models.copy())

        except Exception as e:
            self.error_occurred.emit("general", f"Model loading failed: {str(e)}")


class MainWindow(QMainWindow):
    """Main window for CellSeek GUI application"""

    def __init__(self):
        super().__init__()

        # Initialize project manager
        self.project_manager = ProjectManager()

        # Initialize models (load in background to speed up later operations)
        self.models_loaded = False
        self._models = {}
        self.model_load_worker = None

        # Setup UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_connections()

        # Set window properties
        self.setWindowTitle("CellSeek - Cell Segmentation and Tracking")
        self.setMinimumSize(400, 300)

        # Set window size to 3/4 of screen size
        screen = self.screen().availableGeometry()
        window_width = int(screen.width() * 0.75)
        window_height = int(screen.height() * 0.75)
        self.resize(window_width, window_height)

        # Center window on screen
        self.center_on_screen()

        # Load models in background
        self.load_models_background()

    def setup_ui(self):
        """Setup the main user interface"""
        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left panel - Frame Manager
        self.frame_manager = FrameManagerWidget()
        main_splitter.addWidget(self.frame_manager)

        # Right panel - Tabbed interface for workflow stages
        self.tab_widget = QTabWidget()
        main_splitter.addWidget(self.tab_widget)

        # Create workflow tabs
        self.segmentation_panel = SegmentationPanel()
        self.tracking_panel = TrackingPanel()
        self.analysis_panel = AnalysisPanel()
        self.export_panel = ExportPanel()

        # Add tabs
        self.tab_widget.addTab(self.segmentation_panel, "🔍 Segmentation")
        self.tab_widget.addTab(self.tracking_panel, "🎯 Tracking")
        self.tab_widget.addTab(self.analysis_panel, "📊 Analysis")
        self.tab_widget.addTab(self.export_panel, "💾 Export")

        # Set splitter proportions (25% for frame manager, 75% for tabs)
        # Use percentage-based sizing that adapts to window size
        total_width = 1200  # Default window width
        frame_manager_width = int(total_width * 0.25)  # 25%
        tabs_width = int(total_width * 0.75)  # 75%

        main_splitter.setSizes([frame_manager_width, tabs_width])
        main_splitter.setStretchFactor(0, 0)  # Frame manager doesn't stretch
        main_splitter.setStretchFactor(1, 1)  # Tabs stretch to fill space

        # Set minimum sizes for better layout
        main_splitter.setCollapsible(0, False)  # Don't allow frame manager to collapse
        main_splitter.setCollapsible(1, False)  # Don't allow tabs to collapse

        # Mark setup as completed for resize handling
        self.setup_ui_completed = True

    def setup_menu_bar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # New project action
        new_action = QAction("&New Project", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        # Open project action
        open_action = QAction("&Open Project", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)

        # Save project action
        save_action = QAction("&Save Project", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)

        # Save project as action
        save_as_action = QAction("Save Project &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        # Import frames action
        import_frames_action = QAction("Import &Frames...", self)
        import_frames_action.setShortcut("Ctrl+I")
        import_frames_action.triggered.connect(self.import_frames)
        file_menu.addAction(import_frames_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        # Preferences action
        prefs_action = QAction("&Preferences...", self)
        prefs_action.triggered.connect(self.show_preferences)
        edit_menu.addAction(prefs_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        # Check dependencies action
        check_deps_action = QAction("Check &Dependencies", self)
        check_deps_action.triggered.connect(self.check_dependencies)
        tools_menu.addAction(check_deps_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        # About action
        about_action = QAction("&About CellSeek", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Memory usage label
        self.memory_label = QLabel("Memory: 0 MB")
        self.status_bar.addPermanentWidget(self.memory_label)

        # Timer for memory updates
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(2000)  # Update every 2 seconds

    def setup_connections(self):
        """Setup signal connections between components"""
        # Frame manager connections
        self.frame_manager.frames_loaded.connect(self.on_frames_loaded)
        self.frame_manager.clear_all_requested.connect(self.reset_interface)

        # Segmentation panel connections
        self.segmentation_panel.segmentation_started.connect(
            self.on_segmentation_started
        )
        self.segmentation_panel.segmentation_completed.connect(
            self.on_segmentation_completed
        )
        self.segmentation_panel.segmentation_error.connect(self.on_segmentation_error)
        self.segmentation_panel.annotation_completed.connect(
            self.on_annotation_completed
        )

        # Tracking panel connections
        self.tracking_panel.tracking_started.connect(self.on_tracking_started)
        self.tracking_panel.tracking_completed.connect(self.on_tracking_completed)
        self.tracking_panel.tracking_error.connect(self.on_tracking_error)

        # Analysis panel connections
        self.analysis_panel.analysis_completed.connect(self.on_analysis_completed)

        # Tab change connections
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Set main window reference for panels to access preloaded models
        self.segmentation_panel.set_main_window(self)
        self.tracking_panel.set_main_window(self)

    def center_on_screen(self):
        """Position the window at the center of the screen"""
        screen = self.screen().availableGeometry()
        window = self.frameGeometry()
        # Center both horizontally and vertically
        x = screen.center().x() - window.width() // 2
        y = screen.center().y() - window.height() // 2
        self.move(x, y)

    def load_models_background(self):
        """Load all models in background to speed up later operations"""
        # Start model loading worker after UI is ready
        QTimer.singleShot(1000, self._start_model_loading)

    def _start_model_loading(self):
        """Start the model loading worker thread"""
        if self.model_load_worker is None or not self.model_load_worker.isRunning():
            self.model_load_worker = ModelLoadWorker()

            # Connect worker signals
            self.model_load_worker.progress_update.connect(self._on_model_progress)
            self.model_load_worker.model_loaded.connect(self._on_model_loaded)
            self.model_load_worker.all_models_loaded.connect(self._on_all_models_loaded)
            self.model_load_worker.error_occurred.connect(self._on_model_error)

            # Start loading
            self.model_load_worker.start()

    def _on_model_progress(self, message: str):
        """Handle model loading progress updates"""
        self.status_label.setText(message)

    def _on_model_loaded(self, model_name: str, model_object):
        """Handle individual model loaded"""
        self._models[model_name] = model_object
        print(f"Model loaded: {model_name}")

    def _on_all_models_loaded(self, models: dict):
        """Handle all models loaded"""
        self._models.update(models)
        self.models_loaded = True
        self.status_label.setText("All models loaded - Ready")

        # Pass loaded models to panels
        if hasattr(self, "segmentation_panel"):
            self.segmentation_panel.set_preloaded_models(self._models)
        if hasattr(self, "tracking_panel"):
            self.tracking_panel.set_preloaded_models(self._models)

    def _on_model_error(self, model_name: str, error_message: str):
        """Handle model loading errors"""
        print(f"Model loading error ({model_name}): {error_message}")
        if model_name == "general":
            self.status_label.setText("Ready (model loading failed)")
        else:
            self.status_label.setText("Ready (some models failed to load)")

    def get_preloaded_models(self):
        """Get preloaded models for use by panels"""
        return self._models if self.models_loaded else {}

    # Project management methods
    def new_project(self):
        """Create a new project"""
        if self.project_manager.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save the current project?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Cancel:
                return
            elif reply == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    return

        self.project_manager.new_project()
        self.reset_interface()
        self.update_window_title()
        self.status_label.setText("New project created")

    def open_project(self):
        """Open an existing project"""
        if self.project_manager.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save the current project?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Cancel:
                return
            elif reply == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "CellSeek Project Files (*.csp);;All Files (*)"
        )

        if file_path:
            try:
                self.project_manager.load_project(file_path)
                self.load_project_data()
                self.update_window_title()
                self.status_label.setText(f"Opened project: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to open project:\n{str(e)}"
                )

    def save_project(self) -> bool:
        """Save the current project"""
        if not self.project_manager.project_path:
            return self.save_project_as()

        try:
            project_data = self.collect_project_data()
            self.project_manager.save_project(project_data)
            self.update_window_title()
            self.status_label.setText("Project saved")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{str(e)}")
            return False

    def save_project_as(self) -> bool:
        """Save the current project with a new name"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "CellSeek Project Files (*.csp);;All Files (*)"
        )

        if file_path:
            try:
                project_data = self.collect_project_data()
                self.project_manager.save_project_as(file_path, project_data)
                self.update_window_title()
                self.status_label.setText(f"Project saved as: {Path(file_path).name}")
                return True
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to save project:\n{str(e)}"
                )
                return False
        return False

    def import_frames(self):
        """Import frames into the project"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Frames",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)",
        )

        if file_paths:
            self.frame_manager.load_frames(file_paths)

    def collect_project_data(self) -> Dict[str, Any]:
        """Collect all project data for saving"""
        return {
            "frames": self.frame_manager.get_frame_data(),
            "segmentation": self.segmentation_panel.get_data(),
            "tracking": self.tracking_panel.get_data(),
            "analysis": self.analysis_panel.get_data(),
            "export": self.export_panel.get_data(),
        }

    def load_project_data(self):
        """Load project data into the interface"""
        project_data = self.project_manager.get_project_data()

        if "frames" in project_data:
            self.frame_manager.set_frame_data(project_data["frames"])

        if "segmentation" in project_data:
            self.segmentation_panel.set_data(project_data["segmentation"])

        if "tracking" in project_data:
            self.tracking_panel.set_data(project_data["tracking"])

        if "analysis" in project_data:
            self.analysis_panel.set_data(project_data["analysis"])

        if "export" in project_data:
            self.export_panel.set_data(project_data["export"])

    def reset_interface(self):
        """Reset the interface to initial state"""
        self.frame_manager.clear_frames()
        self.segmentation_panel.reset()
        self.tracking_panel.reset()
        self.analysis_panel.reset()
        self.export_panel.reset()
        self.tab_widget.setCurrentIndex(0)

    def update_window_title(self):
        """Update the window title"""
        title = "CellSeek - Cell Segmentation and Tracking"

        if self.project_manager.project_path:
            project_name = Path(self.project_manager.project_path).stem
            title = f"{title} - {project_name}"

            if self.project_manager.has_unsaved_changes():
                title += "*"

        self.setWindowTitle(title)

    # Event handlers
    def on_frames_loaded(self, frame_count: int):
        """Handle frames loaded event"""
        self.status_label.setText(f"Loaded {frame_count} frames")
        # Enable segmentation tab if frames are loaded
        self.segmentation_panel.setEnabled(frame_count > 0)

        # Pass frame data to tracking panel for later use
        if frame_count > 0:
            # Get all frames and pass them to tracking panel
            all_frames = []
            for i in range(frame_count):
                frame_data = self.frame_manager.get_frame(i)
                if frame_data and "image" in frame_data:
                    all_frames.append(frame_data["image"])

            if all_frames:
                # Just store frames in tracking panel, don't switch tabs yet
                self.tracking_panel.set_all_frames(all_frames)

            # Automatically display the first frame in segmentation panel
            first_frame_data = self.frame_manager.get_frame(0)
            if first_frame_data:
                self.segmentation_panel.set_current_frame(first_frame_data)

    def on_segmentation_started(self):
        """Handle segmentation started event"""
        self.status_label.setText("Running cell segmentation...")
        # Don't show status bar progress - let segmentation panel handle it

    def on_segmentation_completed(self, results):
        """Handle segmentation completed event"""
        self.status_label.setText("Cell segmentation completed")

        # Pass results to tracking panel
        self.tracking_panel.set_segmentation_results(results)
        self.tracking_panel.setEnabled(True)

    def on_annotation_completed(self, updated_masks):
        """Handle SAM annotation completed event"""
        # When SAM annotation updates masks, we need to update the segmentation results
        # and pass them to the tracking panel
        if (
            hasattr(self.segmentation_panel, "segmentation_results")
            and self.segmentation_panel.segmentation_results
        ):
            # Update the tracking panel with the new masks
            updated_results = self.segmentation_panel.segmentation_results.copy()
            self.tracking_panel.set_segmentation_results(updated_results)

            # Update status with mask info
            mask_count = (
                np.max(updated_masks)
                if updated_masks.size > 0 and np.max(updated_masks) > 0
                else 0
            )
            self.status_label.setText(
                f"Segmentation masks updated by SAM annotation ({mask_count} cells)"
            )
        else:
            self.status_label.setText(
                "SAM annotation completed, but no segmentation results to update"
            )

    def on_segmentation_error(self, error_message: str):
        """Handle segmentation error event"""
        self.status_label.setText("Segmentation failed")
        # Don't hide status bar progress - segmentation panel handles it
        QMessageBox.critical(self, "Segmentation Error", error_message)

    def on_tracking_started(self):
        """Handle tracking started event"""
        self.status_label.setText("Running cell tracking...")

    def on_tracking_completed(self, results):
        """Handle tracking completed event"""
        self.status_label.setText("Cell tracking completed")

        # Enable analysis tab
        self.analysis_panel.set_tracking_results(results)
        self.analysis_panel.setEnabled(True)

        # Pass tracking results to export panel
        self.export_panel.set_tracking_results(results)

        # Stay on current tracking tab instead of switching to analysis tab

    def on_tracking_error(self, error_message: str):
        """Handle tracking error event"""
        self.status_label.setText("Tracking failed")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Tracking Error", error_message)

    def on_analysis_completed(self, results):
        """Handle analysis completed event"""
        self.status_label.setText("Analysis completed")

        # Enable export tab
        self.export_panel.set_analysis_results(results)
        self.export_panel.setEnabled(True)

    def on_tab_changed(self, index: int):
        """Handle tab change event"""
        tab_names = ["Segmentation", "Tracking", "Analysis", "Export"]
        if 0 <= index < len(tab_names):
            self.status_label.setText(f"Switched to {tab_names[index]} tab")

    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"Memory: {memory_mb:.0f} MB")
        except ImportError:
            self.memory_label.setText("Memory: N/A")

    # Dialog methods
    def show_preferences(self):
        """Show preferences dialog"""
        # TODO: Implement preferences dialog
        QMessageBox.information(
            self, "Preferences", "Preferences dialog not yet implemented"
        )

    def check_dependencies(self):
        """Check system dependencies"""
        try:
            # Check PyTorch
            import torch

            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()

            # Check other dependencies
            import cv2

            cv2_version = cv2.__version__

            import numpy

            numpy_version = numpy.__version__

            # Try to import SAM and XMem
            try:
                from sam import CellSAM

                sam_available = "✓ Available"
            except ImportError as e:
                sam_available = f"✗ Error: {str(e)}"

            try:
                from xmem import XMem

                xmem_available = "✓ Available"
            except ImportError as e:
                xmem_available = f"✗ Error: {str(e)}"

            message = f"""
Dependency Check Results:

PyTorch: {torch_version}
CUDA Available: {cuda_available}
OpenCV: {cv2_version}
NumPy: {numpy_version}

CellSAM: {sam_available}
XMem: {xmem_available}
            """

            QMessageBox.information(self, "Dependencies", message.strip())

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to check dependencies:\n{str(e)}"
            )

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About CellSeek",
            """
<h3>CellSeek</h3>
<p>Version 1.0.0</p>
<p>A modern GUI application for cell segmentation and tracking using CellSAM and XMem.</p>
<p><b>Features:</b></p>
<ul>
<li>Automatic cell segmentation with CellSAM</li>
<li>Manual annotation with SAM</li>
<li>Cell tracking with XMem</li>
<li>Comprehensive analysis and export tools</li>
</ul>
<p>© 2024 CellSeek Team</p>
            """,
        )

    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up worker thread
        if self.model_load_worker and self.model_load_worker.isRunning():
            self.model_load_worker.quit()
            self.model_load_worker.wait(3000)  # Wait up to 3 seconds

        if self.project_manager.has_unsaved_changes():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            elif reply == QMessageBox.StandardButton.Save:
                if not self.save_project():
                    event.ignore()
                    return

        event.accept()

    def resizeEvent(self, event):
        """Handle window resize to maintain proportional layout"""
        super().resizeEvent(event)

        # Update splitter sizes proportionally when window is resized
        if hasattr(self, "setup_ui_completed"):
            # Find the main splitter
            central_widget = self.centralWidget()
            if central_widget:
                main_layout = central_widget.layout()
                if main_layout and main_layout.count() > 0:
                    main_splitter = main_layout.itemAt(0).widget()
                    if hasattr(main_splitter, "setSizes"):
                        # Calculate new sizes based on current window width
                        total_width = self.width() - 20  # Account for margins
                        frame_manager_width = int(total_width * 0.25)  # 25%
                        tabs_width = int(total_width * 0.75)  # 75%
                        main_splitter.setSizes([frame_manager_width, tabs_width])
