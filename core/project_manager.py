"""
Project manager for handling CellSeek project files
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ProjectManager:
    """Manages CellSeek project loading, saving, and state"""

    def __init__(self):
        self.project_path: Optional[str] = None
        self.project_data: Dict[str, Any] = {}
        self._unsaved_changes = False

        # Create default project structure
        self.new_project()

    def new_project(self):
        """Create a new empty project"""
        self.project_path = None
        self.project_data = {
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "frames": {},
            "segmentation": {},
            "tracking": {},
            "analysis": {},
            "export": {},
            "settings": self._get_default_settings(),
        }
        self._unsaved_changes = False

    def load_project(self, file_path: str):
        """Load a project from file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Project file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.project_data = json.load(f)

            self.project_path = str(file_path)
            self._unsaved_changes = False

            # Validate project data
            self._validate_project_data()

            # Update modification time
            self.project_data["modified"] = datetime.now().isoformat()

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid project file format: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load project: {e}")

    def save_project(self, data: Optional[Dict[str, Any]] = None):
        """Save the current project"""
        if not self.project_path:
            raise ValueError("No project file path set. Use save_project_as() instead.")

        if data:
            self.update_project_data(data)

        self._save_to_file(self.project_path)
        self._unsaved_changes = False

    def save_project_as(self, file_path: str, data: Optional[Dict[str, Any]] = None):
        """Save the project to a new file"""
        if data:
            self.update_project_data(data)

        self.project_path = str(Path(file_path))
        self._save_to_file(self.project_path)
        self._unsaved_changes = False

    def update_project_data(self, data: Dict[str, Any]):
        """Update project data and mark as modified"""
        self.project_data.update(data)
        self.project_data["modified"] = datetime.now().isoformat()
        self._unsaved_changes = True

    def get_project_data(self) -> Dict[str, Any]:
        """Get the current project data"""
        return self.project_data.copy()

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes"""
        return self._unsaved_changes

    def get_project_name(self) -> str:
        """Get the project name"""
        if self.project_path:
            return Path(self.project_path).stem
        return "Untitled Project"

    def get_project_directory(self) -> Optional[Path]:
        """Get the project directory"""
        if self.project_path:
            return Path(self.project_path).parent
        return None

    def create_output_directory(self, subdir: str = "") -> Path:
        """Create and return an output directory for the project"""
        if self.project_path:
            project_dir = Path(self.project_path).parent
            output_dir = project_dir / "output"
        else:
            # Use temporary directory if no project file
            output_dir = Path(tempfile.gettempdir()) / "cellseek_output"

        if subdir:
            output_dir = output_dir / subdir

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def export_project_data(self, file_path: str, format: str = "json"):
        """Export project data in various formats"""
        file_path = Path(file_path)

        if format.lower() == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.project_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def import_project_data(self, file_path: str, format: str = "json"):
        """Import project data from various formats"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Import file not found: {file_path}")

        if format.lower() == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                imported_data = json.load(f)

            # Merge with current project data
            self.project_data.update(imported_data)
            self._unsaved_changes = True
        else:
            raise ValueError(f"Unsupported import format: {format}")

    def backup_project(self) -> Optional[str]:
        """Create a backup of the current project"""
        if not self.project_path:
            return None

        project_path = Path(self.project_path)
        backup_dir = project_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{project_path.stem}_backup_{timestamp}.csp"
        backup_path = backup_dir / backup_name

        try:
            shutil.copy2(self.project_path, backup_path)
            return str(backup_path)
        except Exception as e:
            print(f"Failed to create backup: {e}")
            return None

    def cleanup_backups(self, keep_count: int = 10):
        """Clean up old backup files, keeping only the most recent ones"""
        if not self.project_path:
            return

        project_path = Path(self.project_path)
        backup_dir = project_path.parent / "backups"

        if not backup_dir.exists():
            return

        # Find all backup files for this project
        project_name = project_path.stem
        backup_pattern = f"{project_name}_backup_*.csp"
        backup_files = list(backup_dir.glob(backup_pattern))

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old backups
        for backup_file in backup_files[keep_count:]:
            try:
                backup_file.unlink()
            except Exception as e:
                print(f"Failed to remove backup {backup_file}: {e}")

    def _save_to_file(self, file_path: str):
        """Save project data to file"""
        file_path = Path(file_path)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup before saving
        if file_path.exists():
            self.backup_project()

        # Save to temporary file first, then move to final location
        temp_file = file_path.with_suffix(".tmp")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.project_data, f, indent=2, ensure_ascii=False)

            # Move temporary file to final location
            temp_file.replace(file_path)

            # Clean up old backups
            self.cleanup_backups()

        except Exception as e:
            # Clean up temporary file if it exists
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to save project: {e}")

    def _validate_project_data(self):
        """Validate project data structure"""
        required_keys = [
            "version",
            "frames",
            "segmentation",
            "tracking",
            "analysis",
            "export",
        ]

        for key in required_keys:
            if key not in self.project_data:
                self.project_data[key] = {}

        # Ensure settings exist
        if "settings" not in self.project_data:
            self.project_data["settings"] = self._get_default_settings()

        # Validate version compatibility
        project_version = self.project_data.get("version", "1.0.0")
        if not self._is_version_compatible(project_version):
            raise ValueError(f"Incompatible project version: {project_version}")

    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default project settings"""
        return {
            "cellsam": {
                "diameter": 30.0,
                "flow_threshold": 0.4,
                "cellprob_threshold": 0.0,
            },
            "xmem": {"sam_model": "vit_h", "device": "auto"},
            "analysis": {"time_per_frame": 3.0, "pixel_size": 1.0, "units": "pixels"},
            "export": {"format": "csv", "include_images": True, "include_videos": True},
        }

    def _is_version_compatible(self, version: str) -> bool:
        """Check if project version is compatible"""
        # For now, accept all versions starting with "1."
        return version.startswith("1.")

    def get_setting(self, category: str, key: str, default=None):
        """Get a project setting value"""
        return self.project_data.get("settings", {}).get(category, {}).get(key, default)

    def set_setting(self, category: str, key: str, value):
        """Set a project setting value"""
        if "settings" not in self.project_data:
            self.project_data["settings"] = {}

        if category not in self.project_data["settings"]:
            self.project_data["settings"][category] = {}

        self.project_data["settings"][category][key] = value
        self._unsaved_changes = True
