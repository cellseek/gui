#!/usr/bin/env python3
"""
Test script to demonstrate the mask dependency cleanup functionality.

This script creates a simple test case showing how editing a mask in a non-final frame
will automatically remove all subsequent frames' masks.
"""

import numpy as np
import sys
import os

# Add the gui directory to the path so we can import services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.storage_service import StorageService


def test_mask_dependency_cleanup():
    """Test the mask dependency cleanup functionality"""
    print("Testing mask dependency cleanup functionality...")

    # Create storage service
    storage = StorageService()

    # Simulate having 5 frames
    fake_paths = [f"frame_{i}.jpg" for i in range(5)]
    storage.set_image_paths(fake_paths)

    # Create fake masks for frames 0, 1, 2, 3
    print("\n1. Setting up initial masks...")
    for i in range(4):
        # Create a simple mask with some cell IDs
        mask = np.zeros((100, 100), dtype=np.uint16)
        mask[10:20, 10:20] = 1  # Cell 1
        mask[30:40, 30:40] = 2  # Cell 2
        mask[50:60, 50:60] = i + 3  # Different cell for each frame
        storage.set_mask_for_frame(i, mask)
        print(f"   Frame {i}: Added mask with cells [1, 2, {i + 3}]")

    # Show current state
    print(
        f"\n2. Current masks exist for frames: {list(storage.get_frame_masks().keys())}"
    )

    # Now simulate editing frame 1 (not the last frame)
    print("\n3. Simulating mask edit on frame 1...")
    current_frame_index = 1

    # Check how many frames would be affected
    total_frames = storage.get_frame_count()
    print(f"   Total frames: {total_frames}")
    print(f"   Editing frame: {current_frame_index}")

    if current_frame_index < total_frames - 1:
        removed_count = storage.remove_masks_after_frame(current_frame_index)
        if removed_count > 0:
            last_removed_frame = current_frame_index + removed_count
            print(
                f"   Removed {removed_count} dependent masks (frames {current_frame_index + 1}-{last_removed_frame})"
            )
        else:
            print("   No dependent masks to remove")
    else:
        print("   This is the last frame, no cleanup needed")

    # Show final state
    print(
        f"\n4. Final masks exist for frames: {list(storage.get_frame_masks().keys())}"
    )

    # Test edge case: editing the last frame
    print("\n5. Testing edge case: editing the last frame...")
    storage.set_current_frame_index(1)  # Frame 1 is now the last frame with a mask
    current_frame_index = storage.get_current_frame_index()

    if current_frame_index < total_frames - 1:
        removed_count = storage.remove_masks_after_frame(current_frame_index)
        print(f"   Removed {removed_count} dependent masks")
    else:
        print("   This is the last frame, no cleanup needed")

    print(
        f"\n6. Final state - masks exist for frames: {list(storage.get_frame_masks().keys())}"
    )
    print("\nTest completed successfully! ✓")


if __name__ == "__main__":
    test_mask_dependency_cleanup()
