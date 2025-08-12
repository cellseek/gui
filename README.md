# CellSeek GUI

A modern PyQt6-based graphical user interface for cell segmentation and tracking using CellSAM and XMem.

## Features

- **Modern Dark Theme**: Professional, eye-friendly interface
- **Drag & Drop Frame Loading**: Easy import of image sequences
- **CellSAM Integration**: Automatic cell segmentation on first frame
- **Manual Annotation**: Fine-tune segmentation with SAM tools
- **XMem Tracking**: Track cells across video frames
- **Real-time Analysis**: Cell parameter calculation and visualization
- **Comprehensive Export**: Multiple format support for results

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows/Linux/macOS support

### Quick Setup

1. Navigate to the GUI directory:

```bash
cd gui
```

2. Run the setup script:

```bash
python setup.py
```

3. Start the application:

```bash
python main.py
```

### Manual Installation

If the setup script fails, install dependencies manually:

```bash
pip install PyQt6 numpy opencv-python matplotlib pandas Pillow torch torchvision
```

## Usage

### 1. Project Setup

- Start the application
- Create a new project or open an existing one
- Choose your working directory

### 2. Frame Loading

- Use the Frame Manager tab
- Drag and drop image files or video frames
- Frames will be automatically sorted and displayed as thumbnails
- Supported formats: PNG, JPEG, TIFF, BMP

### 3. Cell Segmentation

- Switch to the Segmentation tab
- Select the first frame for analysis
- Adjust CellSAM parameters:
  - Model type (vit_h, vit_l, vit_b)
  - Flow threshold
  - Cellprob threshold
  - Cell diameter
- Click "Run CellSAM" to perform automatic segmentation
- Masks will be overlaid on the image

### 4. Manual Annotation (Optional)

- Use SAM tools to refine segmentation
- Add or remove cell masks as needed
- Point-based and box-based annotation support

### 5. Cell Tracking

- Move to the Tracking tab
- Configure XMem parameters
- Click "Start Tracking" to track cells across all frames
- Monitor progress in real-time
- View tracking results with overlay visualization

### 6. Analysis

- Access the Analysis tab to view results
- Automatic calculation of cell parameters:
  - Area, perimeter, circularity
  - Mean intensity, standard deviation
  - Centroid coordinates
- Interactive plots and data tables
- Statistical summaries

### 7. Export Results

- Use the Export tab for comprehensive data export
- Multiple format options:
  - **Data**: CSV, Excel, JSON
  - **Images**: PNG, JPEG, TIFF
  - **Videos**: MP4, AVI, MOV
- Batch export capabilities
- File preview and organization

## Project Structure

```
gui/
├── __init__.py              # Main application entry
├── main.py                  # Application launcher
├── setup.py                 # Dependency installer
├── requirements.txt         # Package requirements
├── README.md               # This file
├── main_window.py          # Main GUI window
├── core/
│   └── project_manager.py  # Project file management
└── widgets/
    ├── __init__.py
    ├── frame_manager.py     # Frame loading and display
    ├── segmentation_panel.py # CellSAM integration
    ├── tracking_panel.py    # XMem integration
    ├── analysis_panel.py    # Data analysis and plots
    └── export_panel.py      # Result export
```

## Configuration

### CellSAM Parameters

- **Model Type**: Choose model size vs. accuracy tradeoff
- **Flow Threshold**: Controls flow field sensitivity
- **Cellprob Threshold**: Cell probability cutoff
- **Cell Diameter**: Expected cell size in pixels

### XMem Parameters

- **Memory Bank Size**: Tracking memory capacity
- **Update Frequency**: How often to update tracking
- **Confidence Threshold**: Minimum tracking confidence

### Export Settings

- **Output Directory**: Where to save results
- **File Naming**: Automatic or custom naming schemes
- **Format Options**: Quality and compression settings

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed

   ```bash
   python setup.py
   ```

2. **PyQt6 Installation**: On some systems, use:

   ```bash
   pip install PyQt6 --force-reinstall
   ```

3. **CUDA/GPU Issues**: Ensure PyTorch CUDA compatibility:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Memory Issues**: Reduce batch size or image resolution

### Performance Tips

- Use smaller images for faster processing
- Close unused applications to free memory
- Use GPU acceleration when available
- Save projects regularly to prevent data loss

## Development

### Adding New Features

1. Create new widget in `widgets/` directory
2. Import in `widgets/__init__.py`
3. Add to main window in `main_window.py`
4. Update project manager if persistence needed

### Custom Themes

Modify the stylesheet in `__init__.py` to customize appearance.

## License

This project is part of the CellSeek framework for cell analysis and tracking.

## Support

For issues and questions:

1. Check this README
2. Review error messages in the application
3. Ensure all dependencies are properly installed
4. Check system compatibility

## Changelog

### v1.0.0 (Initial Release)

- Complete GUI framework
- CellSAM and XMem integration
- Modern dark theme
- Comprehensive export system
- Real-time analysis and visualization
