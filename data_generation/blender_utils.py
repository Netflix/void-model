"""
Utility functions for loading HUMOTO data in Blender.
This module provides simplified data loading without PyTorch dependencies.
"""

import os
import pickle
import numpy as np
import json
from pathlib import Path


# Coordinate system conversion matrix (Z-up to Y-up)
Z_UP_TO_Y_UP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])


def quaternion_to_matrix_4x4(quat):
    """
    Convert quaternion to 4x4 transformation matrix.
    Quaternion format: [w, x, y, z, tx, ty, tz] or [w, x, y, z]
    """
    if len(quat) == 7:
        # Full pose: rotation + translation
        w, x, y, z, tx, ty, tz = quat
    else:
        # Rotation only
        w, x, y, z = quat
        tx, ty, tz = 0, 0, 0

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Convert to rotation matrix
    rot = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, tx],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, ty],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, tz],
        [0, 0, 0, 1]
    ])

    return rot


def matrix_to_quaternion(matrix):
    """Convert 4x4 transformation matrix to quaternion [w, x, y, z, tx, ty, tz]."""
    # Extract rotation part
    R = matrix[:3, :3]

    # Extract translation
    t = matrix[:3, 3]

    # Convert rotation matrix to quaternion
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z, t[0], t[1], t[2]])


def load_humoto_pickle(sequence_path, y_up=True):
    """
    Load HUMOTO sequence from pickle file.

    Args:
        sequence_path: Path to sequence directory
        y_up: Convert from Z-up to Y-up coordinate system

    Returns:
        Dictionary with:
        - armature: List of bone transforms per frame
        - objects: Dictionary of object names to pose sequences
        - object_names: List of object names
        - num_frames: Number of frames in sequence
    """
    sequence_name = os.path.basename(sequence_path.rstrip('/'))
    pkl_file = os.path.join(sequence_path, f"{sequence_name}.pkl")

    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Pickle file not found: {pkl_file}")

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Extract basic info
    armature_data = data.get('armature', [])
    objects_data = data.get('objects', {})
    object_names = list(objects_data.keys())

    # Convert coordinate system if needed
    if y_up:
        # Convert object poses to Y-up
        converted_objects = {}
        for obj_name, obj_poses in objects_data.items():
            converted_poses = []
            for pose in obj_poses:
                # Convert pose to matrix, apply transformation, convert back
                pose_matrix = quaternion_to_matrix_4x4(pose)
                pose_matrix_y_up = Z_UP_TO_Y_UP @ pose_matrix
                pose_quat = matrix_to_quaternion(pose_matrix_y_up)
                converted_poses.append(pose_quat)
            converted_objects[obj_name] = np.array(converted_poses)
        objects_data = converted_objects

        # Convert armature data to Y-up
        converted_armature = []
        for frame_bones in armature_data:
            converted_frame = {}
            for bone_name, bone_pose in frame_bones.items():
                pose_matrix = quaternion_to_matrix_4x4(bone_pose)
                pose_matrix_y_up = Z_UP_TO_Y_UP @ pose_matrix
                pose_quat = matrix_to_quaternion(pose_matrix_y_up)
                converted_frame[bone_name] = pose_quat
            converted_armature.append(converted_frame)
        armature_data = converted_armature

    return {
        'armature': armature_data,
        'objects': objects_data,
        'object_names': object_names,
        'num_frames': len(armature_data),
        'sequence_name': sequence_name
    }


def extract_pose_params(armature_data):
    """
    Extract pose parameters organized by bone name.

    Args:
        armature_data: List of frame dictionaries

    Returns:
        Dictionary mapping bone names to arrays of poses [num_frames, 7]
    """
    if len(armature_data) == 0:
        return {}

    # Get all bone names from first frame
    bone_names = list(armature_data[0].keys())
    num_frames = len(armature_data)

    # Organize by bone
    pose_params = {}
    for bone_name in bone_names:
        bone_poses = []
        for frame in armature_data:
            if bone_name in frame:
                bone_poses.append(frame[bone_name])
            else:
                # Use identity if bone missing in frame
                bone_poses.append([1, 0, 0, 0, 0, 0, 0])
        pose_params[bone_name] = np.array(bone_poses)

    return pose_params


def load_object_model(object_name, object_model_dir):
    """
    Load object mesh from model directory.

    Args:
        object_name: Name of the object
        object_model_dir: Path to object models directory

    Returns:
        Dictionary with:
        - vertices: Numpy array of vertices [V, 3]
        - faces: Numpy array of face indices [F, 3]
        - path: Path to the loaded file
    """
    # Try different file formats
    extensions = ['.obj', '.ply', '.stl', '.glb', '.gltf']

    for ext in extensions:
        obj_path = os.path.join(object_model_dir, f"{object_name}{ext}")
        if os.path.exists(obj_path):
            # For now, return path - actual loading will be done by Blender
            return {
                'path': obj_path,
                'format': ext[1:]  # Remove the dot
            }

    # Try finding in subdirectories
    for root, dirs, files in os.walk(object_model_dir):
        for ext in extensions:
            obj_file = f"{object_name}{ext}"
            if obj_file in files:
                obj_path = os.path.join(root, obj_file)
                return {
                    'path': obj_path,
                    'format': ext[1:]
                }

    raise FileNotFoundError(f"Object model not found: {object_name}")


def load_human_model_json(model_path):
    """Load human model definition from JSON file."""
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    return model_data


def subsample_sequence(data, target_frames):
    """
    Subsample sequence data to target number of frames.

    Args:
        data: Dictionary with 'armature' and 'objects'
        target_frames: Target number of frames

    Returns:
        Subsampled data dictionary
    """
    num_frames = data['num_frames']

    if num_frames <= target_frames:
        # No subsampling needed
        return data

    # Calculate indices for even subsampling
    indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)

    # Subsample armature
    armature_subsampled = [data['armature'][i] for i in indices]

    # Subsample objects
    objects_subsampled = {}
    for obj_name, obj_poses in data['objects'].items():
        objects_subsampled[obj_name] = obj_poses[indices]

    return {
        'armature': armature_subsampled,
        'objects': objects_subsampled,
        'object_names': data['object_names'],
        'num_frames': target_frames,
        'sequence_name': data['sequence_name']
    }


def get_light_colors():
    """Get a list of bright, vibrant colors for objects.
    These colors are distinct from dark human colors (dark grey, dark blue, dark purple).
    Expanded color palette for more variety.
    """
    return [
        # Primary bright colors
        (0.9, 0.2, 0.2),    # Bright red
        (0.2, 0.9, 0.2),    # Bright green
        (0.2, 0.6, 0.95),   # Bright sky blue (distinct from dark blue)
        (0.95, 0.95, 0.2),  # Bright yellow

        # Secondary bright colors
        (0.95, 0.2, 0.6),   # Bright pink/magenta (distinct from dark purple)
        (0.95, 0.5, 0.2),   # Bright orange
        (0.2, 0.9, 0.9),    # Bright cyan
        (0.75, 0.2, 0.9),   # Bright violet (distinct from dark purple)

        # Tertiary bright colors
        (0.9, 0.9, 0.45),   # Light yellow-green/lime
        (0.4, 0.95, 0.7),   # Light teal/turquoise
        (0.95, 0.4, 0.4),   # Coral/salmon
        (0.6, 0.9, 0.3),    # Yellow-green

        # Additional vivid colors
        (0.95, 0.7, 0.3),   # Gold/amber
        (0.3, 0.8, 0.95),   # Light blue
        (0.9, 0.3, 0.7),    # Hot pink
        (0.5, 0.95, 0.5),   # Mint green
    ]


def extract_hip_position(armature_data, bone_name='mixamorig:Hips'):
    """
    Extract hip positions from armature data for camera placement.

    Args:
        armature_data: List of frame bone dictionaries
        bone_name: Name of the hip bone

    Returns:
        Numpy array of hip positions [num_frames, 3]
    """
    positions = []
    for frame in armature_data:
        if bone_name in frame:
            pose = frame[bone_name]
            # Translation is last 3 elements of pose
            positions.append(pose[-3:])
        else:
            # Default position
            positions.append([0, 0, 1])
    return np.array(positions)


def calculate_camera_from_hip(hip_position, y_up=True, seed=None, camera_variant='v1'):
    """
    Calculate camera position and look-at point from hip position.
    Camera will be positioned in FRONT of the human, not behind.

    Args:
        hip_position: 3D position of hip [N, 3] or [3]
        y_up: Whether using Y-up coordinate system
        seed: Random seed for variability
        camera_variant: Camera diversity level ('v1', 'v2', 'v3', 'v4', 'v5', 'v6')

    Returns:
        Tuple of (camera_location, look_at_point)
    """
    # Add variability based on seed
    if seed is not None:
        np.random.seed(seed)

    # Randomize distance multiplier (1.0 to 1.4x base distance)
    # This gives good variation without getting too close or too far
    distance_multiplier = np.random.uniform(1.0, 1.4)

    # Randomize height offset based on camera variant
    # v1: 0.3 to 1.3 (increased from 0.5-1.0)
    # v2: 0.2 to 1.6 (increased diversity)
    # v3: 0.1 to 2.0 (more dramatic angles)
    # v4: 0.3 to 1.8 (increased range and diversity)
    # v5: 0.2 to 2.2 (significant increase)
    # v6: 0.0 to 2.8 (maximum diversity with extreme angles)
    height_ranges = {
        'v1': (0.3, 1.3), 'v2': (0.2, 1.6), 'v3': (0.1, 2.0),
        'v4': (0.3, 1.8),
        'v5': (0.2, 2.2),
        'v6': (0.0, 2.8)
    }
    height_min, height_max = height_ranges.get(camera_variant, (0.5, 1.0))
    height_offset = np.random.uniform(height_min, height_max)

    # Randomize downward tilt (0 = straight ahead, positive = looking down)
    # Values 0.0 to 0.5 give subtle to moderate downward angle
    downward_tilt = np.random.uniform(0.0, 0.5)

    # Use average hip position
    if len(hip_position.shape) > 1:
        # Calculate bounds for dynamic distance
        min_hip = np.min(hip_position, axis=0)
        max_hip = np.max(hip_position, axis=0)
        center = (min_hip + max_hip) / 2
        hip_diff = max_hip - min_hip
        hip_diff = np.sqrt(np.sum(hip_diff ** 2))
        # Base distance ensures we see the full scene, multiplier adds variation
        base_dist = hip_diff / 2 + 5.5
        dis = np.clip(base_dist * distance_multiplier, 6.0, 12.0)  # Clamp between 6 and 12 units
    else:
        center = hip_position
        base_dist = 6.5
        dis = np.clip(base_dist * distance_multiplier, 6.0, 12.0)  # Clamp between 6 and 12 units

    # Camera offset - position in FRONT of human
    if y_up:
        # Position camera: slightly to the side, elevated (with variable height), and further away
        camera_height = center[1] + 1.5 + height_offset  # Raised base from 1.2 to 1.5
        camera_loc = np.array([
            center[0] + 1.0,     # Slightly to the side for better view
            camera_height,       # Elevated (eye level + variation)
            center[2] + dis      # POSITIVE Z to be in front
        ])
        # Look at center point at torso height (slightly below camera for better framing)
        # Apply downward tilt - subtracting from Y makes camera look down more
        look_at_height = max(center[1] - downward_tilt, 0.5)  # Don't look below floor level
        look_at = np.array([center[0], look_at_height, center[2]])
    else:
        # Z-up system
        camera_loc = np.array([
            center[0] + 1.0,
            center[1] + dis,     # In front for Z-up
            center[2] + 1.5 + height_offset
        ])
        # Apply downward tilt for Z-up (subtract from Z coordinate)
        look_at_height = max(center[2] - downward_tilt, 0.5)
        look_at = np.array([center[0], center[1], look_at_height])

    return camera_loc, look_at


def get_random_camera_motion(seed=None):
    """
    Randomly select a camera motion type.

    Args:
        seed: Random seed for reproducible selection

    Returns:
        str: One of 'static', 'zoom_in', 'zoom_out', 'circular', 'linear_left', 'linear_right'
    """
    if seed is not None:
        np.random.seed(seed)

    motion_types = ['static', 'zoom_in', 'zoom_out', 'circular', 'linear_left', 'linear_right']
    return np.random.choice(motion_types)


def animate_camera(camera_obj, initial_location, look_at, num_frames, motion_type='static', seed=None, camera_variant='v1'):
    """
    Animate camera with different motion types.

    Args:
        camera_obj: Blender camera object
        initial_location: Initial camera location (x, y, z)
        look_at: Point the camera should look at (x, y, z)
        num_frames: Total number of frames
        motion_type: Type of motion - 'static', 'zoom_in', 'zoom_out', 'circular', 'linear_left', 'linear_right'
        seed: Random seed for reproducible motion
        camera_variant: Camera diversity level ('v1', 'v2', 'v3', 'v4', 'v5', 'v6')
    """
    import bpy
    import math

    if seed is not None:
        np.random.seed(seed)

    # Clear existing animation data
    if camera_obj.animation_data:
        camera_obj.animation_data_clear()

    if motion_type == 'static':
        # No animation - camera stays in place
        return

    elif motion_type == 'zoom_in':
        # Zoom towards the look_at point
        # Convert to numpy arrays to avoid Blender Vector issues
        initial_loc = np.array([initial_location[0], initial_location[1], initial_location[2]])
        look_at_loc = np.array([look_at[0], look_at[1], look_at[2]])

        direction = look_at_loc - initial_loc
        distance = np.linalg.norm(direction)
        direction_normalized = direction / distance

        # Zoom in amount based on camera variant
        # v1: 35-55% closer (increased from 30-45%)
        # v2: 40-65% closer (more movement)
        # v3: 45-72% closer (significant zoom)
        # v4: 40-70% closer (increased movement)
        # v5: 50-80% closer (significant increase)
        # v6: 60-85% closer (maximum zoom with dramatic effect)
        zoom_ranges = {
            'v1': (0.35, 0.55), 'v2': (0.40, 0.65), 'v3': (0.45, 0.72),
            'v4': (0.40, 0.70),
            'v5': (0.50, 0.80),
            'v6': (0.60, 0.85)
        }
        zoom_min, zoom_max = zoom_ranges.get(camera_variant, (0.3, 0.45))
        zoom_amount = np.random.uniform(zoom_min, zoom_max) * distance
        final_location = initial_loc + direction_normalized * zoom_amount

        # Set keyframes (convert back to tuple for Blender)
        camera_obj.location = tuple(initial_loc)
        camera_obj.keyframe_insert(data_path="location", frame=1)

        camera_obj.location = tuple(final_location)
        camera_obj.keyframe_insert(data_path="location", frame=num_frames)

        # Smooth interpolation
        if camera_obj.animation_data and camera_obj.animation_data.action:
            for fcurve in camera_obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'BEZIER'
                    keyframe.handle_left_type = 'AUTO'
                    keyframe.handle_right_type = 'AUTO'

    elif motion_type == 'zoom_out':
        # Zoom away from the look_at point
        # Convert to numpy arrays to avoid Blender Vector issues
        initial_loc = np.array([initial_location[0], initial_location[1], initial_location[2]])
        look_at_loc = np.array([look_at[0], look_at[1], look_at[2]])

        direction = initial_loc - look_at_loc
        distance = np.linalg.norm(direction)
        direction_normalized = direction / distance

        # Zoom out amount based on camera variant
        # v1: 15-30% further (increased from 10-20%)
        # v2: 20-40% further (more movement)
        # v3: 25-50% further (significant zoom out)
        # v4: 20-45% further (increased movement)
        # v5: 30-60% further (significant increase)
        # v6: 40-75% further (maximum zoom out with wide framing)
        zoom_ranges = {
            'v1': (0.15, 0.30), 'v2': (0.20, 0.40), 'v3': (0.25, 0.50),
            'v4': (0.20, 0.45),
            'v5': (0.30, 0.60),
            'v6': (0.40, 0.75)
        }
        zoom_min, zoom_max = zoom_ranges.get(camera_variant, (0.1, 0.2))
        zoom_amount = np.random.uniform(zoom_min, zoom_max) * distance
        final_location = initial_loc + direction_normalized * zoom_amount

        # Set keyframes (convert back to tuple for Blender)
        camera_obj.location = tuple(initial_loc)
        camera_obj.keyframe_insert(data_path="location", frame=1)

        camera_obj.location = tuple(final_location)
        camera_obj.keyframe_insert(data_path="location", frame=num_frames)

        # Smooth interpolation
        if camera_obj.animation_data and camera_obj.animation_data.action:
            for fcurve in camera_obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'BEZIER'
                    keyframe.handle_left_type = 'AUTO'
                    keyframe.handle_right_type = 'AUTO'

    elif motion_type == 'circular':
        # Circular motion around the look_at point
        # Convert to numpy arrays to avoid Blender Vector issues
        initial_loc = np.array([initial_location[0], initial_location[1], initial_location[2]])
        look_at_loc = np.array([look_at[0], look_at[1], look_at[2]])

        # Calculate orbit parameters
        direction = initial_loc - look_at_loc
        radius = np.linalg.norm(direction[:2])  # Horizontal radius
        height = initial_loc[2]

        # Determine rotation direction (clockwise or counter-clockwise)
        clockwise = np.random.choice([True, False])

        # Determine arc angle based on camera variant
        # v1: 60-110 degrees (increased from 45-90)
        # v2: 70-130 degrees (more rotation)
        # v3: 80-150 degrees (significant arc)
        # v4: 75-140 degrees (increased rotation)
        # v5: 90-180 degrees (significant increase - half circle possible)
        # v6: 120-220 degrees (maximum rotation with dramatic sweeps)
        arc_ranges = {
            'v1': (60, 110), 'v2': (70, 130), 'v3': (80, 150),
            'v4': (75, 140),
            'v5': (90, 180),
            'v6': (120, 220)
        }
        arc_min, arc_max = arc_ranges.get(camera_variant, (45, 90))
        arc_angle = np.random.uniform(math.radians(arc_min), math.radians(arc_max))
        if clockwise:
            arc_angle = -arc_angle

        # Calculate initial angle
        initial_angle = math.atan2(direction[1], direction[0])

        # Create keyframes for smooth circular motion
        num_keyframes = min(10, num_frames)  # Use up to 10 keyframes for smooth motion
        for i in range(num_keyframes + 1):
            frame = 1 + int(i * (num_frames - 1) / num_keyframes)
            progress = i / num_keyframes

            # Calculate angle for this frame
            angle = initial_angle + arc_angle * progress

            # Calculate position
            x = look_at_loc[0] + radius * math.cos(angle)
            y = look_at_loc[1] + radius * math.sin(angle)
            z = height

            camera_obj.location = (x, y, z)
            camera_obj.keyframe_insert(data_path="location", frame=frame)

        # Smooth interpolation
        if camera_obj.animation_data and camera_obj.animation_data.action:
            for fcurve in camera_obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'BEZIER'
                    keyframe.handle_left_type = 'AUTO'
                    keyframe.handle_right_type = 'AUTO'

    elif motion_type == 'linear_left':
        # Linear motion to the left (negative X direction)
        # Movement amount based on camera variant
        # v1: 2.0-4.0 units (increased from 1.5-3.0)
        # v2: 2.5-5.0 units (more movement)
        # v3: 3.0-6.0 units (significant panning)
        # v4: 2.5-5.5 units (increased movement)
        # v5: 3.5-7.0 units (significant increase)
        # v6: 4.5-9.0 units (maximum movement with dramatic panning)
        movement_ranges = {
            'v1': (2.0, 4.0), 'v2': (2.5, 5.0), 'v3': (3.0, 6.0),
            'v4': (2.5, 5.5),
            'v5': (3.5, 7.0),
            'v6': (4.5, 9.0)
        }
        move_min, move_max = movement_ranges.get(camera_variant, (1.5, 3.0))
        total_movement = np.random.uniform(move_min, move_max)

        # Set keyframes
        camera_obj.location = initial_location
        camera_obj.keyframe_insert(data_path="location", frame=1)

        final_location = (
            initial_location[0] - total_movement,
            initial_location[1],
            initial_location[2]
        )
        camera_obj.location = final_location
        camera_obj.keyframe_insert(data_path="location", frame=num_frames)

        # Smooth interpolation
        if camera_obj.animation_data and camera_obj.animation_data.action:
            for fcurve in camera_obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'BEZIER'
                    keyframe.handle_left_type = 'AUTO'
                    keyframe.handle_right_type = 'AUTO'

    elif motion_type == 'linear_right':
        # Linear motion to the right (positive X direction)
        # Movement amount based on camera variant
        # v1: 2.0-4.0 units (increased from 1.5-3.0)
        # v2: 2.5-5.0 units (more movement)
        # v3: 3.0-6.0 units (significant panning)
        # v4: 2.5-5.5 units (increased movement)
        # v5: 3.5-7.0 units (significant increase)
        # v6: 4.5-9.0 units (maximum movement with dramatic panning)
        movement_ranges = {
            'v1': (2.0, 4.0), 'v2': (2.5, 5.0), 'v3': (3.0, 6.0),
            'v4': (2.5, 5.5),
            'v5': (3.5, 7.0),
            'v6': (4.5, 9.0)
        }
        move_min, move_max = movement_ranges.get(camera_variant, (1.5, 3.0))
        total_movement = np.random.uniform(move_min, move_max)

        # Set keyframes
        camera_obj.location = initial_location
        camera_obj.keyframe_insert(data_path="location", frame=1)

        final_location = (
            initial_location[0] + total_movement,
            initial_location[1],
            initial_location[2]
        )
        camera_obj.location = final_location
        camera_obj.keyframe_insert(data_path="location", frame=num_frames)

        # Smooth interpolation
        if camera_obj.animation_data and camera_obj.animation_data.action:
            for fcurve in camera_obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'BEZIER'
                    keyframe.handle_left_type = 'AUTO'
                    keyframe.handle_right_type = 'AUTO'


if __name__ == "__main__":
    # Test loading
    import sys
    if len(sys.argv) > 1:
        seq_path = sys.argv[1]
        print(f"Loading sequence: {seq_path}")
        data = load_humoto_pickle(seq_path, y_up=True)
        print(f"Loaded {data['num_frames']} frames")
        print(f"Objects: {data['object_names']}")
        pose_params = extract_pose_params(data['armature'])
        print(f"Bones: {list(pose_params.keys())[:5]}...")
