#!/usr/bin/env python3
"""
Generate first frame images for all scenarios to enable manual physics review.

This script renders the first frame (with human) for all sequences that have
pre-computed vertices, saving them for manual review to determine which objects
should not have physics applied.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
import json


def find_sequences_with_verts(data_dir):
    """Find all sequences with pre-computed vertices."""
    sequences = []
    data_path = Path(data_dir)

    for seq_dir in sorted(data_path.iterdir()):
        if not seq_dir.is_dir():
            continue

        seq_name = seq_dir.name
        verts_file = seq_dir / f"{seq_name}_human_verts.pkl"

        if verts_file.exists():
            # Also load the YAML to get object list
            yaml_file = seq_dir / f"{seq_name}.yaml"
            if yaml_file.exists():
                with open(yaml_file, 'r') as f:
                    metadata = yaml.safe_load(f)
                    sequences.append({
                        'name': seq_name,
                        'path': str(seq_dir),
                        'objects': metadata.get('objects', [])
                    })

    return sequences


def render_first_frame_simple(seq_name, data_dir, objects_dir, output_path):
    """Render first frame using a simple Blender script."""

    blender_script = f"""
import bpy
import sys
import os

# Add current directory to path
sys.path.insert(0, '{os.getcwd()}')

# Import render module
import render_paired_videos_blender as render_module
from blender_utils import *

# Parse minimal args
class Args:
    dataset_dir = '{data_dir}'
    object_model = '{objects_dir}'
    y_up = True
    resolution_x = 800
    resolution_y = 600
    fps = 12
    target_frames = 1
    seed = 42
    floor_texture = None
    wall_texture = None
    object_texture = None
    add_walls = True
    enable_physics = False
    random_human_color = False
    rename_output = None

args = Args()

# Clear scene
render_module.clear_scene()

# Setup scene
scene = render_module.setup_scene(args.resolution_x, args.resolution_y, args.fps)

# Load sequence
seq_path = os.path.join(args.dataset_dir, '{seq_name}')
pkl_path = os.path.join(seq_path, '{seq_name}.pkl')

try:
    # Load data
    data = render_module.load_humoto_pickle(pkl_path)
    data_sub = render_module.subsample_sequence(data, 1, 'uniform')

    # Create ground
    ground = render_module.create_ground_plane(size=10, y_up=True, texture_path=None)

    # Load walls
    walls = render_module.load_walls(size=10, height=3, y_up=True, texture_path=None)

    # Load objects
    objects_data = render_module.load_and_setup_objects(
        data_sub['objects'],
        args.object_model,
        y_up=True,
        texture_path=None
    )

    # Create human
    human_obj = render_module.create_human_from_verts(
        seq_path,
        '{seq_name}',
        1,
        y_up=True,
        random_color=False
    )

    # Get hip position for camera
    hip_pos = render_module.extract_pose_params(data_sub['human']['poses'][0])['hip_position']

    # Setup camera
    camera = render_module.setup_camera(hip_pos, y_up=True)

    # Animate objects (just frame 1)
    object_pose_params = {{k: v['poses'][:1] for k, v in data_sub['objects'].items()}}
    render_module.animate_objects(objects_data, object_pose_params, 1, y_up=True)

    # Animate human
    render_module.animate_human(human_obj, seq_path, '{seq_name}', 1, y_up=True)

    # Setup lighting
    render_module.setup_lighting(y_up=True)

    # Render frame 1
    scene.frame_set(1)
    scene.render.filepath = '{output_path}'
    bpy.ops.render.render(write_still=True)

    print("RENDER_SUCCESS")

except Exception as e:
    print(f"RENDER_ERROR: {{e}}")
    import traceback
    traceback.print_exc()
"""

    # Write script to temp file
    temp_script = Path("_temp_render_script.py")
    with open(temp_script, 'w') as f:
        f.write(blender_script)

    try:
        # Run Blender in background mode
        cmd = f"""blender --background --python {temp_script}"""
        result = subprocess.run(
            ['bash', '-c', cmd],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd()
        )

        # Check for success
        if 'RENDER_SUCCESS' in result.stdout and Path(output_path).exists():
            return True
        else:
            # Print detailed error info for debugging
            if 'RENDER_ERROR' in result.stdout:
                # Extract error
                for line in result.stdout.split('\n'):
                    if 'RENDER_ERROR' in line or 'ERROR' in line or 'Traceback' in line:
                        print(f"      {line}")
            else:
                # Print last few lines of output for debugging
                lines = result.stdout.split('\n')
                print(f"      Last output lines:")
                for line in lines[-10:]:
                    if line.strip():
                        print(f"        {line}")

                # Also check stderr
                if result.stderr:
                    print(f"      Stderr:")
                    for line in result.stderr.split('\n')[-5:]:
                        if line.strip():
                            print(f"        {line}")
            return False

    except subprocess.TimeoutExpired:
        print(f"      Timeout")
        return False
    except Exception as e:
        print(f"      Exception: {e}")
        return False
    finally:
        # Clean up temp script
        if temp_script.exists():
            temp_script.unlink()


def main():
    # Paths
    data_dir = "./humoto_release/humoto_0805"
    objects_dir = "./humoto_release/humoto_objects_0805"
    output_dir = "./physics_review_frames"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all sequences
    print("Finding sequences with pre-computed vertices...")
    sequences = find_sequences_with_verts(data_dir)
    print(f"Found {len(sequences)} sequences")

    # Save sequence metadata for the UI
    metadata_path = Path(output_dir) / "sequences_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(sequences, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Render first frames
    print("\nRendering first frames...")
    successful = 0
    failed = 0

    for i, seq_info in enumerate(sequences, 1):
        seq_name = seq_info['name']
        output_path = Path(output_dir) / f"{seq_name}.png"

        # Skip if already exists
        if output_path.exists():
            print(f"[{i}/{len(sequences)}] {seq_name} - already exists, skipping")
            successful += 1
            continue

        print(f"[{i}/{len(sequences)}] {seq_name}")
        if render_first_frame_simple(seq_name, data_dir, objects_dir, str(output_path)):
            print(f"    ✓ Success")
            successful += 1
        else:
            print(f"    ✗ Failed")
            failed += 1

    print("\n" + "="*50)
    print("Generation Complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
