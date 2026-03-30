#!/usr/bin/env python3
"""
Generate physics configuration file for all scenarios.
This creates a JSON file listing all objects in each sequence.
You can then edit it to mark which objects should be ACTIVE (move) or PASSIVE (static).
"""

import os
import json
import pickle
from pathlib import Path

def main():
    data_dir = "./humoto_release/humoto_0805"
    output_file = "./physics_config.json"

    print("Scanning sequences for objects...")

    config = {
        "_instructions": {
            "description": "Physics configuration for each sequence",
            "how_to_use": [
                "1. Find your sequence name in this file",
                "2. For each object, set 'physics' to either:",
                "   - 'active': object will move/fall when human removed",
                "   - 'passive': object stays static (furniture, walls, floor)",
                "3. Save this file",
                "4. Run generate_2500_scenarios.sh - it will use your settings"
            ],
            "default_if_not_specified": "active (objects will move)"
        },
        "sequences": {}
    }

    sequences_found = 0

    # Scan all sequence directories
    for seq_dir in sorted(Path(data_dir).iterdir()):
        if not seq_dir.is_dir():
            continue

        seq_name = seq_dir.name

        # Check if this sequence has pre-computed vertices
        verts_file = seq_dir / f"{seq_name}_human_verts.pkl"
        if not verts_file.exists():
            continue

        # Load the pickle file to get object information
        pkl_file = seq_dir / f"{seq_name}.pkl"
        if not pkl_file.exists():
            continue

        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # Extract object names
            if 'objects' in data and isinstance(data['objects'], dict):
                objects = list(data['objects'].keys())
            else:
                # No objects found
                objects = []

            if objects:
                sequences_found += 1
                config["sequences"][seq_name] = {
                    "objects": {
                        obj: {
                            "physics": "active",  # Default: will move
                            "notes": ""  # You can add notes here
                        }
                        for obj in objects
                    }
                }

                print(f"  {seq_name}: {len(objects)} objects - {', '.join(objects)}")

        except Exception as e:
            print(f"  WARNING: Could not read {seq_name}: {e}")
            continue

    # Save configuration
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Found {sequences_found} sequences with pre-computed vertices")
    print(f"✓ Configuration saved to: {output_file}")
    print(f"{'='*60}")
    print("\nNEXT STEPS:")
    print("1. Open physics_config.json in a text editor")
    print("2. For each sequence, review the objects")
    print("3. Change 'physics' from 'active' to 'passive' for:")
    print("   - Furniture (tables, chairs, shelves)")
    print("   - Large static objects")
    print("   - Objects that should NOT move")
    print("4. Keep 'active' for objects that should fall/move")
    print("5. Save the file")
    print(f"\nExample entry:")
    print(json.dumps({
        "sequence_name": {
            "objects": {
                "table": {"physics": "passive", "notes": "large furniture"},
                "mug": {"physics": "active", "notes": "small prop"}
            }
        }
    }, indent=2))

if __name__ == "__main__":
    main()
