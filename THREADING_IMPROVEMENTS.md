# Threading Improvements - UI Freeze Fix

## Problem

The UI was freezing during expensive operations because all AI model processing (SAM, CellSAM, CUTIE) was running synchronously on the main thread. This caused:

- UI becoming unresponsive during segmentation/tracking
- Poor user experience with blocked interactions
- Application appearing to hang during processing

## Solution

Moved all expensive operations to background threads using PyQt6's QThread framework with proper signal-slot communication.

## Changes Made

### 1. SAM Service (`services/sam_service.py`)

**Before:**

- Direct synchronous calls to `predict_point()` and `predict_box()`
- UI froze during segmentation

**After:**

- Inherited from `QObject` for signal support
- Added async methods: `predict_point_async()`, `predict_box_async()`
- Results delivered via signals: `sam_result_ready`, `sam_error`
- Status updates via `status_update` signal

### 2. SAM Worker (`workers/sam_worker.py`)

**Before:**

- Inherited from `QThread` but used synchronously
- Methods called directly blocking the main thread

**After:**

- Properly implemented `run()` method for background execution
- Added async entry points: `set_image_async()`, `predict_point_async()`, `predict_box_async()`
- Emits signals: `result_ready(mask, score)`, `error_occurred(str)`, `status_update(str)`
- Task queue system for handling different operations

### 3. CellSAM Service (`services/cellsam_service.py`)

**Before:**

- Synchronous `segment_first_frame()` blocking UI during first frame processing

**After:**

- Inherited from `QObject` for signal support
- Added `segment_first_frame_async()` method
- Results delivered via `segmentation_complete(masks)` signal
- Error handling via `segmentation_error(str)` signal
- Kept legacy sync method for backwards compatibility

### 4. CellSAM Worker (`workers/cellsam_worker.py`)

**Before:**

- Simple class without threading capabilities

**After:**

- Inherited from `QThread`
- Added `run_async()` entry point
- Proper `run()` implementation for background execution
- Emits signals: `result_ready(masks)`, `error_occurred(str)`, `status_update(str)`
- Backwards compatible with sync calls

### 5. CUTIE Service (`services/cutie_service.py`)

**Before:**

- Synchronous `track()` method blocking UI during frame tracking

**After:**

- Inherited from `QObject` for signal support
- Added `track_async()` method
- Results delivered via `tracking_complete(mask)` signal
- Error handling via `tracking_error(str)` signal
- Kept legacy sync method for backwards compatibility

### 6. CUTIE Worker (`workers/cutie_worker.py`)

**Before:**

- Simple class without threading capabilities

**After:**

- Inherited from `QThread`
- Added `track_async()` entry point
- Proper `run()` implementation for background execution
- Emits signals: `result_ready(mask)`, `error_occurred(str)`, `status_update(str)`
- Backwards compatible with sync calls

### 7. Frame-by-Frame Widget (`widgets/frame_by_frame_widget.py`)

**Before:**

- Direct synchronous calls to tracking methods
- UI froze during `next_frame()` tracking operations

**After:**

- Added `setup_async_connections()` method
- Connected to async signals from CUTIE service
- Implemented `_on_tracking_complete()` and `_on_tracking_error()` handlers
- Uses `track_async()` instead of blocking `track()`
- Proper state management with `_pending_frame_index`

### 8. Main Window (`main_window.py`)

**Before:**

- Synchronous CellSAM processing on frame import
- UI blocked during first frame segmentation

**After:**

- Connected to CellSAM async signals in `setup_connections()`
- Implemented `_on_cellsam_complete()` and `_on_cellsam_error()` handlers
- Uses `segment_first_frame_async()` instead of blocking call
- Proper state management with `_pending_frame_paths`

## Benefits

### User Experience

- ✅ UI remains responsive during all operations
- ✅ Real-time status updates during processing
- ✅ Users can interact with other parts of the application
- ✅ Visual feedback shows processing progress

### Technical Benefits

- ✅ Proper separation of UI and computation threads
- ✅ Non-blocking operations for all expensive tasks
- ✅ Signal-slot architecture for clean async communication
- ✅ Error handling doesn't block the UI
- ✅ Backwards compatibility maintained

### Performance

- ✅ AI models run at full speed without UI overhead
- ✅ Background processing utilizes available CPU/GPU resources
- ✅ Multiple operations can be queued properly
- ✅ Better resource management

## Migration Guide

### For Developers

If you were using the old synchronous methods:

```python
# OLD - blocks UI
mask = sam_service.sam_worker.predict_point(point)

# NEW - async with signals
sam_service.on_point_clicked(point)
# Result comes via sam_result_ready signal
```

```python
# OLD - blocks UI
masks = cellsam_service.segment_first_frame(path)

# NEW - async with signals
cellsam_service.segment_first_frame_async(path)
# Result comes via segmentation_complete signal
```

```python
# OLD - blocks UI
mask = cutie_service.track(prev_img, prev_mask, curr_img)

# NEW - async with signals
cutie_service.track_async(prev_img, prev_mask, curr_img)
# Result comes via tracking_complete signal
```

### Signal Connections

All services now emit these signals:

- `result_ready` / `segmentation_complete` / `tracking_complete` - Success
- `error_occurred` / `segmentation_error` / `tracking_error` - Errors
- `status_update` - Progress updates

Connect to these signals in your UI components to handle async results.

## Testing

- ✅ All imports successful
- ✅ No syntax errors in updated files
- ✅ Backwards compatibility maintained
- ✅ Signal-slot connections properly established

The UI should now remain responsive during all expensive AI operations while providing real-time feedback to users.
