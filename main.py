#!/usr/bin/env python3
"""
CellSeek GUI Application Entry Point

This script launches the CellSeek GUI application for cell segmentation and tracking.
"""

import sys


def main():
    """Main entry point for the CellSeek GUI application"""
    try:
        # Import and run the GUI application from the __init__ module
        from __init__ import main as gui_main

        gui_main()

    except ImportError as e:
        print(f"Failed to import GUI modules: {e}")
        print("\nPlease ensure PyQt6 is installed:")
        print("pip install PyQt6")
        sys.exit(1)

    except Exception as e:
        print(f"Failed to start CellSeek GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
