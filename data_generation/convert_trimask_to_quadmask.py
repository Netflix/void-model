#!/usr/bin/env python3
"""
Convert tri-mask videos to quad-mask videos by detecting overlap areas.

Tri-mask values:
  0   = remove object (black)
  127 = modify area (grey)
  255 = keep unchanged (white)

Quad-mask values:
  0   = pure remove (black, no difference between full and removed)
  63  = overlap (black mask + actual difference)
  127 = modify area (grey)
  255 = keep unchanged (white)
"""

import os
import sys
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


def quantize_to_trimask(mask):
    """Quantize mask to tri-mask values (0, 127, 255)."""
    trimask = np.zeros_like(mask)
    trimask[mask < 100] = 0
    trimask[(mask >= 100) & (mask <= 150)] = 127
    trimask[mask > 150] = 255
    return trimask


def create_quadmask(mask_frame, rgb_full_frame, rgb_removed_frame, diff_threshold=10):
    """
    Create quad-mask from tri-mask and RGB frames.

    Args:
        mask_frame: Original mask frame (numpy array)
        rgb_full_frame: Full RGB frame (numpy array)
        rgb_removed_frame: RGB frame with object removed (numpy array)
        diff_threshold: Threshold for detecting significant differences

    Returns:
        Quad-mask numpy array
    """
    # Step 1: Quantize to tri-mask
    trimask = quantize_to_trimask(mask_frame)

    # Step 2: Compute grey (modify) area from rgb difference
    diff = np.abs(rgb_full_frame.astype(float) - rgb_removed_frame.astype(float))
    diff_gray = np.mean(diff, axis=2)
    grey_from_diff = (diff_gray > diff_threshold)

    # Get black mask (human/object)
    is_black = (trimask == 0)

    # Step 3: Detect overlap and separate regions
    # Overlap = where BOTH black mask AND grey (difference) exist
    overlap = is_black & grey_from_diff

    # Pure grey = grey areas NOT under black mask
    pure_grey = grey_from_diff & ~is_black

    # Pure black = black areas with NO grey underneath
    pure_black = is_black & ~grey_from_diff

    # Step 4: Create quad-mask
    quadmask = np.ones_like(mask_frame) * 255  # Start with all white (keep)

    # Set pure grey areas (modification regions NOT under object)
    quadmask[pure_grey] = 127

    # Set pure black areas (object with no underlying change)
    quadmask[pure_black] = 0

    # Set overlap areas (object with underlying change)
    overlap_value = 63  # 127 // 2
    quadmask[overlap] = overlap_value

    return quadmask


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


def create_video_from_frames(frame_dir, output_path, fps=12, lossless=True):
    """Create video from frames using ffmpeg."""
    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")

    if lossless:
        # Use lossless grayscale FFV1 codec to preserve exact pixel values
        # This avoids color space conversion issues with H.264
        cmd = [
            "ffmpeg", "-framerate", str(fps),
            "-i", frame_pattern,
            "-vf", "format=gray",
            "-c:v", "ffv1", "-level", "3",
            output_path,
            "-y", "-loglevel", "error"
        ]
    else:
        # Standard lossy encoding
        cmd = [
            "ffmpeg", "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_path,
            "-y", "-loglevel", "error"
        ]
    subprocess.run(cmd, check=True)


def process_video_folder(folder_path, diff_threshold=10, fps=12):
    """
    Process a single folder containing rgb_full.mp4, rgb_removed.mp4, and mask.mp4.

    Creates a new quadmask.mp4 file.
    """
    folder_path = Path(folder_path)

    # Check required files (support both .mp4 and .mkv)
    mask_video = folder_path / "mask.mp4" if (folder_path / "mask.mp4").exists() else folder_path / "mask.mkv"
    rgb_full_video = folder_path / "rgb_full.mp4" if (folder_path / "rgb_full.mp4").exists() else folder_path / "rgb_full.mkv"
    rgb_removed_video = folder_path / "rgb_removed.mp4" if (folder_path / "rgb_removed.mp4").exists() else folder_path / "rgb_removed.mkv"

    if not all([mask_video.exists(), rgb_full_video.exists(), rgb_removed_video.exists()]):
        print(f"Skipping {folder_path.name}: missing required videos")
        return False

    # Create temp directories for frames
    temp_dir = folder_path / "temp_quadmask_conversion"
    temp_dir.mkdir(exist_ok=True)

    mask_frames_dir = temp_dir / "mask_frames"
    rgb_full_frames_dir = temp_dir / "rgb_full_frames"
    rgb_removed_frames_dir = temp_dir / "rgb_removed_frames"
    quadmask_frames_dir = temp_dir / "quadmask_frames"

    for d in [mask_frames_dir, rgb_full_frames_dir, rgb_removed_frames_dir, quadmask_frames_dir]:
        d.mkdir(exist_ok=True)

    try:
        print(f"\nProcessing {folder_path.name}...")

        # Extract frames
        print("  Extracting frames...")
        num_frames_mask = extract_frames(str(mask_video), str(mask_frames_dir))
        num_frames_full = extract_frames(str(rgb_full_video), str(rgb_full_frames_dir))
        num_frames_removed = extract_frames(str(rgb_removed_video), str(rgb_removed_frames_dir))

        if not (num_frames_mask == num_frames_full == num_frames_removed):
            print(f"  WARNING: Frame count mismatch: mask={num_frames_mask}, full={num_frames_full}, removed={num_frames_removed}")

        num_frames = min(num_frames_mask, num_frames_full, num_frames_removed)

        # Process each frame
        print(f"  Creating quad-masks for {num_frames} frames...")
        stats = {0: 0, 63: 0, 127: 0, 255: 0}  # Accumulate stats

        for i in tqdm(range(1, num_frames + 1), desc="  Frames"):
            # Load frames
            mask_frame = np.array(Image.open(mask_frames_dir / f"frame_{i:06d}.png").convert('L'))
            rgb_full_frame = np.array(Image.open(rgb_full_frames_dir / f"frame_{i:06d}.png").convert('RGB'))
            rgb_removed_frame = np.array(Image.open(rgb_removed_frames_dir / f"frame_{i:06d}.png").convert('RGB'))

            # Create quad-mask
            quadmask = create_quadmask(mask_frame, rgb_full_frame, rgb_removed_frame, diff_threshold)

            # Accumulate stats
            unique, counts = np.unique(quadmask, return_counts=True)
            for val, count in zip(unique, counts):
                if val in stats:
                    stats[val] += count

            # Save quad-mask frame
            quadmask_img = Image.fromarray(quadmask)
            quadmask_img.save(quadmask_frames_dir / f"frame_{i:06d}.png")

        # Create video from quad-mask frames
        print("  Creating quad-mask video (lossless grayscale)...")
        output_video = folder_path / "quadmask.mkv"
        create_video_from_frames(str(quadmask_frames_dir), str(output_video), fps, lossless=True)

        # Print statistics
        total_pixels = sum(stats.values())
        print(f"  Quad-mask statistics:")
        print(f"    Value 0 (pure remove): {stats[0]:,} pixels ({stats[0]/total_pixels*100:.2f}%)")
        print(f"    Value 63 (overlap): {stats[63]:,} pixels ({stats[63]/total_pixels*100:.2f}%)")
        print(f"    Value 127 (modify): {stats[127]:,} pixels ({stats[127]/total_pixels*100:.2f}%)")
        print(f"    Value 255 (keep): {stats[255]:,} pixels ({stats[255]/total_pixels*100:.2f}%)")

        print(f"  ✓ Created {output_video}")

        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"  ✗ Error processing {folder_path.name}: {e}")
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False


def find_video_folders(root_dir):
    """Find all folders containing the required video files."""
    root_dir = Path(root_dir)
    folders = []

    for folder in root_dir.iterdir():
        if not folder.is_dir():
            continue

        # Check for .mp4 or .mkv files
        mask_video = folder / "mask.mp4" if (folder / "mask.mp4").exists() else folder / "mask.mkv"
        rgb_full_video = folder / "rgb_full.mp4" if (folder / "rgb_full.mp4").exists() else folder / "rgb_full.mkv"
        rgb_removed_video = folder / "rgb_removed.mp4" if (folder / "rgb_removed.mp4").exists() else folder / "rgb_removed.mkv"

        if all([mask_video.exists(), rgb_full_video.exists(), rgb_removed_video.exists()]):
            folders.append(folder)

    return sorted(folders)


def main():
    parser = argparse.ArgumentParser(description="Convert tri-mask videos to quad-mask videos")
    parser.add_argument("input_dir", nargs="?", default=".",
                        help="Directory containing video folders (default: current directory)")
    parser.add_argument("--folder", type=str,
                        help="Process only a specific folder")
    parser.add_argument("--diff-threshold", type=int, default=10,
                        help="Threshold for detecting differences (default: 10)")
    parser.add_argument("--fps", type=int, default=12,
                        help="Output video FPS (default: 12)")
    args = parser.parse_args()

    if args.folder:
        # Process single folder
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"Error: Folder {folder_path} does not exist")
            sys.exit(1)

        success = process_video_folder(folder_path, args.diff_threshold, args.fps)
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
            print("Looking for folders with: mask.mp4, rgb_full.mp4, rgb_removed.mp4")
            sys.exit(1)

        print(f"Found {len(folders)} video folders to process")

        success_count = 0
        for folder in folders:
            if process_video_folder(folder, args.diff_threshold, args.fps):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Processed {success_count}/{len(folders)} folders successfully")


if __name__ == "__main__":
    main()
