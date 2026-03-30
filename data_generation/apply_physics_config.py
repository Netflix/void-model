#!/usr/bin/env python3
"""
Helper module to load and apply physics configuration.
This is imported by render_paired_videos_blender.py
"""

import json
import os

_physics_config = None

def load_physics_config(config_file="./physics_config.json"):
    """Load physics configuration from JSON file."""
    global _physics_config

    if _physics_config is not None:
        return _physics_config

    if not os.path.exists(config_file):
        print(f"  No physics config found at {config_file}, using defaults")
        _physics_config = {}
        return _physics_config

    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
            _physics_config = data.get("sequences", {})
            print(f"  Loaded physics config for {len(_physics_config)} sequences")
            return _physics_config
    except Exception as e:
        print(f"  WARNING: Could not load physics config: {e}")
        _physics_config = {}
        return _physics_config

def get_object_physics_type(sequence_name, object_name):
    """
    Get physics type for a specific object in a sequence.

    Returns:
        'active': object should move/fall
        'passive': object should stay static
    """
    config = load_physics_config()

    if sequence_name not in config:
        # Default: all objects are active
        return 'active'

    seq_config = config[sequence_name]
    objects = seq_config.get("objects", {})

    if object_name not in objects:
        # Default: active
        return 'active'

    obj_config = objects[object_name]
    physics_type = obj_config.get("physics", "active")

    # Validate
    if physics_type not in ['active', 'passive']:
        print(f"    WARNING: Invalid physics type '{physics_type}' for {object_name}, using 'active'")
        return 'active'

    return physics_type

def print_physics_summary(sequence_name, objects_list):
    """Print a summary of physics settings for a sequence."""
    config = load_physics_config()

    if sequence_name not in config:
        print(f"  Using default physics (all active) for {sequence_name}")
        return

    active_count = 0
    passive_count = 0

    for obj_name in objects_list:
        physics_type = get_object_physics_type(sequence_name, obj_name)
        if physics_type == 'active':
            active_count += 1
        else:
            passive_count += 1

    print(f"  Physics config: {active_count} ACTIVE, {passive_count} PASSIVE")
