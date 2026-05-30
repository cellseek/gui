# CellSeek

Desktop GUI for **frame-by-frame cell segmentation and tracking** on microscopy image sequences and videos. Built with **PyQt6**, it combines automatic models (Cellpose-SAM, Trackastra) with interactive refinement (Segment Anything).

## Features

- **Import** image sequences or a single video (drag-and-drop or file picker)
- **Preprocess** before tracking: crop, brightness/contrast/gamma, trim frame range; preview and scrub on video
- **Segment & track** one frame at a time:
  - First frame: automatic segmentation with **Cellpose-SAM** (cpsam)
  - Later frames: press **Next** to auto-segment and link identities from the previous frame (**CellSAM + Trackastra**)
  - Manual correction with **SAM** (click, box, or paint prompts) and brush tools; undo/redo mask history
- **Relink** cell IDs on the current frame after edits (keeps mask shapes, reassigns labels via Trackastra)
- **Export** per-frame tracking CSV, cell summary CSV, annotated video, and labeled frame images with morphological metrics

## Workflow

```
Import → Preprocess → Track (frame-by-frame) → Export
```

1. **Import** — Drop images or one video. Videos are decoded lazily (no full export to disk). Only one video per import.
2. **Preprocess** — Adjust tone, crop ROI, and trim the frame range; confirm to start tracking.
3. **Track** — Review/edit masks on each frame; use **Next (D)** to advance. Empty next frames are segmented and linked automatically when the current frame has a mask.
4. **Export** — Open export from the tracking screen when ready.

Use **Restart** to return to preprocess (e.g. change crop or trim) without re-importing.

## Supported formats

| Type   | Extensions                          |
| ------ | ----------------------------------- |
| Images | PNG, JPG, JPEG, TIFF, TIF, BMP, GIF |
| Video  | MP4, AVI, MOV, MKV, WMV, FLV, WEBM  |

Videos are subsampled by default (one logical frame every **30** source frames, ~1 Hz at 30 fps). Image sequences use natural sort order (`natsort`).

## Requirements

- **Python 3** (version compatible with PyTorch and dependencies below)
- **GPU recommended** — SAM uses CUDA when available; Trackastra uses `device="automatic"`. CPU fallback is possible but slower.
- Disk space for model weights (see [Model weights](#model-weights))

### Python dependencies

Install from `requirements.txt`:

```
PyQt6
cellpose
natsort
numpy
opencv-python
psutil
segment-anything
torch
trackastra
```

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the app

From the project root (`gui/`):

```bash
python main.py
```

On startup the app:

1. Ensures `weights/sam_vit_b_01ec64.pth` exists (downloads from Meta if missing)
2. Launches the dark-themed main window

**Cellpose-SAM (cpsam)** weights download on first segmentation if not present under `weights/cpsam` (see `utils/cellsam_segment.py`).

Optional: place `resources/icon.png` for the window icon.

## Model weights

| Model                   | Purpose                                | Location / source                                             |
| ----------------------- | -------------------------------------- | ------------------------------------------------------------- |
| SAM ViT-B               | Interactive click/box/paint refinement | `weights/sam_vit_b_01ec64.pth` — auto-downloaded by `main.py` |
| Cellpose-SAM (cpsam)    | Initial frame + advance segmentation   | `weights/cpsam` or Cellpose download on first use             |
| Trackastra `general_2d` | Identity linking between frames        | Downloaded by Trackastra on first use                         |

`weights/` is gitignored; weights are fetched at runtime.

## Keyboard shortcuts (tracking screen)

| Key                     | Action                                                   |
| ----------------------- | -------------------------------------------------------- |
| **A**                   | Previous frame                                           |
| **D**                   | Next frame (auto-segment/link if next frame has no mask) |
| **Tab**                 | Cycle annotation tool mode                               |
| **Ctrl+Z** / **Ctrl+Y** | Undo / redo mask edits                                   |

Tool modes: View, Click Add, Box Add, Paint Add (SAM), Mask Remove, Edit Cell ID (also selectable via radio buttons).

## Project structure

```
gui/
├── main.py                 # Entry point, theme, SAM weight check
├── main_window.py          # Stacked workflow: import → preprocess → track → export
├── requirements.txt
├── weights/                # Runtime model files (not in git)
├── services/               # CellSAM, SAM, tracking, storage, export, annotations
├── widgets/                # UI screens and interactive canvas
├── workers/                # Background threads for models and video import
└── utils/                  # Segmentation, tracking, preprocess, media helpers
```

## Export outputs

The export screen can generate:

- **Per-frame CSV** — tracking data with morphological and movement metrics
- **Cell summary CSV** — aggregated statistics per cell ID
- **Annotated video** — masks overlaid on source frames
- **Individual frames** — labeled images per time point

Configure time-per-frame (minutes) for movement calculations in the export UI (`ExportService` defaults to 3 minutes per frame until changed).

## Notes

- Processing resizes frames for inference (max dimension **512** for segmentation paths) while preserving workflow masks as label images (`uint16`).
- Status bar shows process **memory usage** (updated every 2 s).
- If CellSAM fails on the first frame, you can still annotate manually with SAM or paint tools.
- Editing a frame and advancing may clear masks on **later** frames after relink/advance operations that change identities downstream.
