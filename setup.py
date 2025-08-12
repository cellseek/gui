#!/usr/bin/env python3
"""
CellSeek GUI Setup Script

This script helps set up the CellSeek GUI application by installing dependencies
and checking system requirements.
"""

import importlib
import subprocess
import sys


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    return True


def install_package(package_name, import_name=None):
    """Install a package using pip"""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} already installed")
        return True
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            print(f"✓ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package_name}")
            return False


def main():
    """Main setup function"""
    print("CellSeek GUI Setup")
    print("==================")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Required packages
    packages = [
        ("PyQt6", "PyQt6"),
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("Pillow", "PIL"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ]

    failed_packages = []

    for package, import_name in packages:
        if not install_package(package, import_name):
            failed_packages.append(package)

    if failed_packages:
        print("\nFailed to install packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease install them manually using:")
        print(f"pip install {' '.join(failed_packages)}")
        sys.exit(1)

    print("\n✓ All dependencies installed successfully!")
    print("\nTo run the CellSeek GUI:")
    print("python main.py")


if __name__ == "__main__":
    main()
