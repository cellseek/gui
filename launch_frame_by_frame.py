#!/usr/bin/env python3
"""
Launch script for the new frame-by-frame CellSeek GUI
"""

import sys
from pathlib import Path

# Add the gui directory to Python path
gui_dir = Path(__file__).parent
sys.path.insert(0, str(gui_dir))

# Launch the application
if __name__ == "__main__":
    from main_frame_by_frame import main

    main()
