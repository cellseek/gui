# CellSeek Frame-by-Frame GUI

A new interface for frame-by-frame cell segmentation and tracking using CellSAM, SAM, and CUTIE.

## Overview

This GUI implements a frame-by-frame workflow where users are involved in every step of the cell tracking process. It's designed for scenarios where manual correction and supervision are needed at each frame.

## Workflow

### 1. Media Import Screen

- **Drag and Drop**: Support for video files (.mp4, .avi, .mov, etc.) or multiple image files (.png, .jpg, .tiff, etc.)
- **File Browser**: Manual selection via file dialogs
- **Video Processing**: Automatic frame extraction from video files with user-configurable frame step
- **Validation**: Checks that all imported files are valid images

### 2. Frame-by-Frame Processing Screen

#### Layout

- **Left Panel**: Previous frame (initially empty for the first frame)
- **Right Panel**: Current frame (editable)
- **Control Panel**: Frame navigation and tools
- **Progress Panel**: Status and progress indicators

#### Navigation

- **Previous Frame Button** (A key): Go to previous frame
- **Next Frame Button** (D key): Go to next frame
- **Frame Counter**: Shows current frame number and total frames

#### Segmentation Tools

##### Auto Segmentation

- **CellSAM Integration**: Automatic cell segmentation for the first frame
- **Configurable Parameters**:
  - Diameter: Expected cell diameter in pixels
  - Flow Threshold: Segmentation sensitivity
  - Cell Probability Threshold: Detection confidence

##### Manual Annotation (SAM)

- **View Mode** (1 key): Browse without editing
- **Add by Click** (2 key): Click to add single cell segments
- **Add by Box** (3 key): Draw bounding box to add segments
- **Remove Mask** (4 key): Click on existing masks to remove them

#### Keyboard Shortcuts

- **A**: Previous frame
- **D**: Next frame
- **S**: Run auto segmentation (CellSAM)
- **1-4**: Switch between annotation modes
- **Escape**: Return to import screen (with confirmation)

## Technical Features

### CellSAM Integration

- Automatic cell segmentation using the CellSAM model
- Adjustable parameters for different cell types and imaging conditions
- Progress tracking with cancellation support

### SAM Integration

- Interactive segmentation for manual corrections
- Point-based and box-based prompting
- Real-time preview and mask overlay
- Efficient mask management and ID assignment

### CUTIE Integration (Planned)

- Frame-by-frame tracking using the modified CUTIE model
- Each frame uses the corrected masks from the previous frame
- Support for manual corrections at every step
- Memory-efficient processing suitable for long sequences

### User Interface

- **Dark Theme**: Modern dark interface optimized for long work sessions
- **Responsive Layout**: Resizable panels that adapt to window size
- **Visual Feedback**: Clear indication of current mode and available actions
- **Progress Tracking**: Real-time status updates and progress bars
- **Memory Monitoring**: Display of current memory usage

## Installation and Usage

### Prerequisites

```bash
pip install PyQt6 opencv-python numpy torch torchvision
pip install natsort psutil  # Optional but recommended
```

### Running the Application

```bash
cd gui
python launch_frame_by_frame.py
```

Or directly:

```bash
cd gui
python main_frame_by_frame.py
```

## File Structure

```
gui/
├── main_frame_by_frame.py          # New main application entry point
├── launch_frame_by_frame.py        # Launch script
├── new_main_window.py              # New main window implementation
├── widgets/
│   ├── media_import_widget.py      # Video/image import with drag & drop
│   ├── frame_by_frame_widget.py    # Main frame-by-frame interface
│   └── __init__.py                 # Widget exports
└── README_FRAME_BY_FRAME.md        # This file
```

## Key Improvements

### Over Previous Interface

1. **Simplified Workflow**: Clear two-stage process (import → process)
2. **Better Visual Layout**: Side-by-side frame comparison
3. **Enhanced Interaction**: Intuitive drag & drop and keyboard shortcuts
4. **Real-time Feedback**: Immediate visual feedback for all operations
5. **Flexible Input**: Support for both video and image sequences

### User Experience

1. **Guided Process**: Clear indication of next steps
2. **Error Handling**: Graceful handling of file errors and processing failures
3. **Progress Tracking**: Visual progress indicators for long operations
4. **Memory Awareness**: Memory usage monitoring to prevent system overload

## Future Enhancements

1. **CUTIE Integration**: Complete frame-by-frame tracking implementation
2. **Batch Processing**: Support for processing multiple sequences
3. **Export Options**: Various output formats for masks and tracking data
4. **Undo/Redo**: History management for manual corrections
5. **Advanced Visualization**: Enhanced overlay options and mask visualization
6. **Settings Panel**: Persistent user preferences and model parameters

## Troubleshooting

### Common Issues

1. **Memory Usage**: Large videos may consume significant memory during frame extraction
2. **File Formats**: Some video codecs may not be supported by OpenCV
3. **Model Loading**: CellSAM and SAM models require adequate GPU memory or may fall back to CPU

### Performance Tips

1. **Frame Step**: Use frame step > 1 for videos to reduce the number of frames
2. **Image Size**: Consider resizing very large images for better performance
3. **GPU Memory**: Monitor GPU memory usage when processing large sequences
