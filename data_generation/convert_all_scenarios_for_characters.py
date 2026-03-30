#!/usr/bin/env python3
"""
Convert all HUMOTO scenarios for both Remy and Sophie characters.
This creates parallel datasets with character-specific animations.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Paths
HUMOTO_DIR = "./humoto_release/humoto_0805"
OUTPUT_BASE = "./humoto_characters_converted"
REMY_FBX = "./human_model/Remy_mixamo_bone.fbx"
SOPHIE_FBX = "./human_model/Sophie_mixamo_bone.fbx"

# Character info
CHARACTERS = {
    "remy": {"fbx": REMY_FBX, "output_dir": f"{OUTPUT_BASE}/remy"},
    "sophie": {"fbx": SOPHIE_FBX, "output_dir": f"{OUTPUT_BASE}/sophie"}
}

# Progress tracking
PROGRESS_FILE = "./character_conversion_progress.json"


def get_all_sequences():
    """Get all sequence directories."""
    sequences = []
    for item in sorted(os.listdir(HUMOTO_DIR)):
        seq_path = os.path.join(HUMOTO_DIR, item)
        if os.path.isdir(seq_path):
            pkl_file = os.path.join(seq_path, f"{item}.pkl")
            if os.path.exists(pkl_file):
                sequences.append(item)
    return sequences


def load_progress():
    """Load conversion progress."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"remy": {}, "sophie": {}}


def save_progress(progress):
    """Save conversion progress."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def run_blender_script(script_path, args):
    """Run a Blender Python script with arguments."""
    cmd = ["blender", "--background", "--python", script_path, "--"] + args

    # Check if blender command exists, if not use python directly with bpy
    if subprocess.run(["which", "blender"], capture_output=True).returncode != 0:
        # Use python with bpy import instead
        cmd = ["python", "-c", f"""
import sys
sys.argv = {['script'] + args}
exec(open('{script_path}').read())
"""]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def convert_sequence_for_character(sequence_name, character_name, character_fbx, output_base):
    """
    Convert one sequence for a specific character.
    Runs the 4-step conversion process.
    """
    print(f"  Converting {sequence_name} for {character_name}...")

    original_path = os.path.join(HUMOTO_DIR, sequence_name)

    # Step directories
    step1_dir = f"{output_base}_step1/{sequence_name}"
    step2_dir = f"{output_base}_step2/{sequence_name}"
    step3_dir = f"{output_base}_step3/{sequence_name}"
    final_dir = os.path.join(output_base, sequence_name)

    os.makedirs(step1_dir, exist_ok=True)
    os.makedirs(step2_dir, exist_ok=True)
    os.makedirs(step3_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Step 1: Clear human scale
    print(f"    [1/4] Clearing scale...")
    success, stdout, stderr = run_blender_script(
        "scripts/clear_human_scale.py",
        ["--input_dir", original_path, "--output_dir", step1_dir]
    )
    if not success:
        print(f"    ERROR in step 1: {stderr[:200]}")
        return False

    # Step 2: Transfer character model
    print(f"    [2/4] Transferring {character_name} model...")
    success, stdout, stderr = run_blender_script(
        "scripts/transfer_human_model.py",
        ["--input_dir", step1_dir, "--output_dir", step2_dir, "--human_model", character_fbx]
    )
    if not success:
        print(f"    ERROR in step 2: {stderr[:200]}")
        return False

    # Step 3: Extract pickle data
    print(f"    [3/4] Extracting pickle data...")
    success, stdout, stderr = run_blender_script(
        "scripts/extract_pk_data.py",
        ["--input_dir", step2_dir, "--output_dir", step3_dir]
    )
    if not success:
        print(f"    ERROR in step 3: {stderr[:200]}")
        return False

    # Step 4: Copy metadata
    print(f"    [4/4] Copying metadata...")
    result = subprocess.run(
        ["python3", "scripts/copy_text.py", "--src", original_path, "--dst", final_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"    ERROR in step 4: {result.stderr[:200]}")
        return False

    # Copy final converted files from step3 to final directory
    for item in os.listdir(step3_dir):
        src = os.path.join(step3_dir, item)
        dst = os.path.join(final_dir, item)
        if os.path.isfile(src):
            subprocess.run(["cp", src, dst])

    print(f"    ✓ Completed!")
    return True


def main():
    print("="*60)
    print("Converting All Scenarios for Remy & Sophie")
    print("="*60)

    # Get all sequences
    sequences = get_all_sequences()
    print(f"\nFound {len(sequences)} sequences to convert")

    # Load progress
    progress = load_progress()

    # Convert for each character
    for char_name, char_info in CHARACTERS.items():
        print(f"\n{'='*60}")
        print(f"Converting for {char_name.upper()}")
        print(f"{'='*60}")

        char_fbx = char_info["fbx"]
        char_output = char_info["output_dir"]

        if not os.path.exists(char_fbx):
            print(f"ERROR: Character FBX not found: {char_fbx}")
            continue

        os.makedirs(char_output, exist_ok=True)

        completed = 0
        failed = 0
        skipped = 0

        for idx, seq in enumerate(sequences, 1):
            # Check if already completed
            if progress.get(char_name, {}).get(seq) == "completed":
                print(f"[{idx}/{len(sequences)}] {seq}: SKIPPED (already completed)")
                skipped += 1
                continue

            print(f"[{idx}/{len(sequences)}] {seq}")

            success = convert_sequence_for_character(seq, char_name, char_fbx, char_output)

            if success:
                progress.setdefault(char_name, {})[seq] = "completed"
                completed += 1
            else:
                progress.setdefault(char_name, {})[seq] = "failed"
                failed += 1

            # Save progress every 10 sequences
            if idx % 10 == 0:
                save_progress(progress)
                print(f"\n  Progress: {completed} completed, {failed} failed, {skipped} skipped")

        # Final save
        save_progress(progress)

        print(f"\n{char_name.upper()} Conversion Summary:")
        print(f"  Completed: {completed}")
        print(f"  Failed: {failed}")
        print(f"  Skipped: {skipped}")

    # Overall summary
    print("\n" + "="*60)
    print("Overall Conversion Summary")
    print("="*60)

    for char_name in CHARACTERS.keys():
        char_progress = progress.get(char_name, {})
        completed = sum(1 for v in char_progress.values() if v == "completed")
        failed = sum(1 for v in char_progress.values() if v == "failed")

        print(f"{char_name.upper()}:")
        print(f"  Completed: {completed}/{len(sequences)}")
        print(f"  Failed: {failed}/{len(sequences)}")

    print("="*60)
    print(f"Converted datasets saved to: {OUTPUT_BASE}/")
    print(f"Progress tracking: {PROGRESS_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()
