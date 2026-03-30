#!/usr/bin/env python3
"""
Convert quad masks to hybrid grid-aligned masks for VLM inference.

Approach:
1. Grid-align white (255) and grey (127): cell gets 127 if it has ANY 127, else 255
2. Keep black (0) pixel-accurate (non-gridded)
3. Compute overlap (63) where pixel-accurate black overlaps with grid-aligned 127
"""

import os
import sys
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


def get_video_dimensions(video_path):
    """Get video width and height using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width, height = map(int, result.stdout.strip().split('x'))
    return width, height


def calculate_grid_size(width, height, min_cells=8):
    """
    Calculate grid size maintaining aspect ratio with minimum cells on shorter side.

    Args:
        width: Video width in pixels
        height: Video height in pixels
        min_cells: Minimum number of cells on shorter dimension

    Returns:
        (grid_cols, grid_rows): Number of grid cells in x and y directions
    """
    aspect_ratio = width / height

    if width <= height:
        # Width is shorter or equal
        grid_cols = min_cells
        grid_rows = max(min_cells, round(min_cells / aspect_ratio))
    else:
        # Height is shorter
        grid_rows = min_cells
        grid_cols = max(min_cells, round(min_cells * aspect_ratio))

    return grid_cols, grid_rows


def convert_mask_to_hybrid_grid(mask, grid_cols, grid_rows):
    """
    Convert quad mask to hybrid grid-aligned mask.

    Process:
    1. Grid-align 127 (grey): cell gets 127 if ANY pixel is 127, else 255
    2. Keep 63 (overlap) pixel-perfect (non-gridded)
    3. Apply pixel-accurate black (0) on top
    4. Recompute overlap (63) where pixel-accurate black overlaps grid-aligned 127

    Args:
        mask: 2D numpy array with quad mask values (0, 63, 127, 255)
        grid_cols: Number of grid columns
        grid_rows: Number of grid rows

    Returns:
        Hybrid grid-aligned mask as 2D numpy array
    """
    height, width = mask.shape
    cell_height = height / grid_rows
    cell_width = width / grid_cols

    # Step 1: Extract pixel-accurate masks (will be applied later)
    black_mask = (mask == 0)
    overlap_mask = (mask == 63)

    # Step 2: Create grid-aligned background with 127 only
    grid_background = np.ones((height, width), dtype=np.uint8) * 255

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Get cell boundaries
            y1 = int(row * cell_height)
            y2 = int((row + 1) * cell_height)
            x1 = int(col * cell_width)
            x2 = int((col + 1) * cell_width)

            # Extract cell values
            cell_values = mask[y1:y2, x1:x2]

            # Check if cell has ANY grey (127)
            has_grey = np.any(cell_values == 127)

            if has_grey:
                grid_background[y1:y2, x1:x2] = 127

    # Step 3: Start with grid background
    output_mask = grid_background.copy()

    # Step 4: Apply pixel-accurate black
    output_mask[black_mask] = 0

    # Step 5: Compute new overlap where black meets grid-aligned 127
    # (black overlapping with 127 becomes 63)
    overlap_from_127 = black_mask & (grid_background == 127)
    output_mask[overlap_from_127] = 63

    # Step 6: Apply pixel-accurate 63 from original mask on top
    output_mask[overlap_mask] = 63

    return output_mask


def extract_frames(video_path, output_dir):
    """Extract all frames from a video using ffmpeg."""
    output_pattern = os.path.join(output_dir, "frame_%06d.png")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vsync", "0",
        output_pattern,
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

    # Count frames
    frames = sorted([f for f in os.listdir(output_dir) if f.startswith("frame_")])
    return len(frames)


def create_video_from_frames(frame_dir, output_path, fps=12):
    """Create grayscale video from frames using ffmpeg."""
    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")

    # Use H.264 with yuv444p to avoid chroma subsampling that corrupts mask values
    # crf=0 for lossless, yuv444p to preserve exact pixel values
    cmd = [
        "ffmpeg", "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
        output_path,
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)


def process_video_folder(folder_path, min_cells=8, fps=12, visualize=False):
    """
    Process a single folder containing mask video.

    Creates a new mask_grid.mkv file with hybrid grid-aligned masks.
    """
    folder_path = Path(folder_path)

    # Find mask video (try both .mp4 and .mkv)
    mask_video = None
    for ext in ['.mp4', '.mkv']:
        # Try quadmask first, then mask
        for name in ['quadmask', 'mask']:
            candidate = folder_path / f"{name}{ext}"
            if candidate.exists():
                mask_video = candidate
                break
        if mask_video:
            break

    if not mask_video or not mask_video.exists():
        print(f"Skipping {folder_path.name}: no mask video found")
        return False

    print(f"\nProcessing {folder_path.name}...")
    print(f"  Using mask: {mask_video.name}")

    # Get video dimensions
    width, height = get_video_dimensions(mask_video)
    grid_cols, grid_rows = calculate_grid_size(width, height, min_cells)

    print(f"  Video: {width}x{height}, Grid: {grid_cols}x{grid_rows} cells")
    print(f"  Cell size: ~{width/grid_cols:.1f}x{height/grid_rows:.1f} pixels")
    print(f"  Method: Hybrid (grid 127/255, pixel-accurate 0 and 63)")

    # Create temp directories
    temp_dir = folder_path / "temp_grid_conversion"
    temp_dir.mkdir(exist_ok=True)

    mask_frames_dir = temp_dir / "mask_frames"
    grid_frames_dir = temp_dir / "grid_frames"

    for d in [mask_frames_dir, grid_frames_dir]:
        d.mkdir(exist_ok=True)

    try:
        # Extract frames
        print("  Extracting frames...")
        num_frames = extract_frames(str(mask_video), str(mask_frames_dir))

        # Process each frame
        print(f"  Converting {num_frames} frames to hybrid grid masks...")
        stats_original = {0: 0, 63: 0, 127: 0, 255: 0}
        stats_grid = {0: 0, 63: 0, 127: 0, 255: 0}

        for i in tqdm(range(1, num_frames + 1), desc="  Frames"):
            # Load mask frame
            mask_frame = np.array(Image.open(mask_frames_dir / f"frame_{i:06d}.png").convert('L'))

            # Accumulate original stats
            unique, counts = np.unique(mask_frame, return_counts=True)
            for val, count in zip(unique, counts):
                if val in stats_original:
                    stats_original[val] += count

            # Convert to hybrid grid mask
            grid_mask = convert_mask_to_hybrid_grid(mask_frame, grid_cols, grid_rows)

            # Accumulate grid stats
            unique, counts = np.unique(grid_mask, return_counts=True)
            for val, count in zip(unique, counts):
                if val in stats_grid:
                    stats_grid[val] += count

            # Save grid mask frame
            grid_mask_img = Image.fromarray(grid_mask)
            grid_mask_img.save(grid_frames_dir / f"frame_{i:06d}.png")

            # Optional: save visualization of first frame
            if visualize and i == 1:
                vis_dir = folder_path / "visualization"
                vis_dir.mkdir(exist_ok=True)

                # Save original
                Image.fromarray(mask_frame).save(vis_dir / "original_mask_frame1.png")

                # Save grid version
                grid_mask_img.save(vis_dir / "grid_mask_frame1.png")

                print(f"  Saved visualization to {vis_dir}/")

        # Create video from grid mask frames
        print("  Creating hybrid grid mask video...")
        output_video = folder_path / "mask_grid.mp4"
        create_video_from_frames(str(grid_frames_dir), str(output_video), fps)

        # Print statistics comparison
        total_pixels = sum(stats_original.values())
        print(f"\n  Original mask statistics:")
        for val in [0, 63, 127, 255]:
            pct = stats_original[val]/total_pixels*100
            print(f"    Value {val:3d}: {stats_original[val]:,} pixels ({pct:.2f}%)")

        print(f"\n  Hybrid grid mask statistics:")
        for val in [0, 63, 127, 255]:
            pct = stats_grid[val]/total_pixels*100
            change = stats_grid[val] - stats_original[val]
            change_sign = "+" if change > 0 else ""
            print(f"    Value {val:3d}: {stats_grid[val]:,} pixels ({pct:.2f}%) [{change_sign}{change:,}]")

        print(f"\n  ✓ Created {output_video}")

        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"  ✗ Error processing {folder_path.name}: {e}")
        import traceback
        traceback.print_exc()
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False


def find_video_folders(root_dir):
    """Find all folders containing mask videos."""
    root_dir = Path(root_dir)
    folders = []

    for folder in root_dir.iterdir():
        if not folder.is_dir():
            continue

        # Check for mask video (quadmask or mask, .mp4 or .mkv)
        has_mask = False
        for ext in ['.mp4', '.mkv']:
            for name in ['quadmask', 'mask']:
                if (folder / f"{name}{ext}").exists():
                    has_mask = True
                    break
            if has_mask:
                break

        if has_mask:
            # Skip if mask_grid.mp4 already exists
            if not (folder / "mask_grid.mp4").exists():
                folders.append(folder)

    return sorted(folders)


def main():
    parser = argparse.ArgumentParser(
        description="Convert quad masks to hybrid grid-aligned masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hybrid approach:
  1. Grid-align white (255) and grey (127) - cell gets 127 if ANY pixel is 127
  2. Keep black (0) and overlap (63) pixel-accurate (non-gridded)
  3. Compute additional overlap (63) where black overlaps with grid-aligned 127

This matches VLM inference (coarse grey regions) while preserving precise removal and overlap masks.

Examples:
  # Process single folder with visualization
  python convert_masks_to_grid_hybrid.py --folder ./my_video --visualize

  # Process all folders with 16x16 minimum grid
  python convert_masks_to_grid_hybrid.py --min-cells 16

  # Process specific test folder
  python convert_masks_to_grid_hybrid.py --folder ./500_remy_videos_v3/carry_mixing_bowl_left_hand_walk_around-229_v3_r2 --visualize
        """
    )
    parser.add_argument("input_dir", nargs="?", default=".",
                        help="Directory containing video folders (default: current directory)")
    parser.add_argument("--folder", type=str,
                        help="Process only a specific folder")
    parser.add_argument("--min-cells", type=int, default=8,
                        help="Minimum cells on shorter dimension (default: 8)")
    parser.add_argument("--fps", type=int, default=12,
                        help="Output video FPS (default: 12)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization of first frame (original vs grid)")
    args = parser.parse_args()

    if args.folder:
        # Process single folder
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"Error: Folder {folder_path} does not exist")
            sys.exit(1)

        success = process_video_folder(folder_path, args.min_cells, args.fps, args.visualize)
        sys.exit(0 if success else 1)

    else:
        # Process all folders in input directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory {input_dir} does not exist")
            sys.exit(1)

        folders = find_video_folders(input_dir)

        if not folders:
            print(f"No video folders found in {input_dir}")
            print("Looking for folders with: mask.mp4/mkv or quadmask.mp4/mkv")
            sys.exit(1)

        print(f"Found {len(folders)} video folders to process")
        print(f"Grid config: min {args.min_cells} cells on shorter side")
        print(f"Method: Hybrid (grid 127/255, pixel-accurate 0 and 63)")

        success_count = 0
        for folder in folders:
            if process_video_folder(folder, args.min_cells, args.fps, args.visualize):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Processed {success_count}/{len(folders)} folders successfully")


if __name__ == "__main__":
    main()
