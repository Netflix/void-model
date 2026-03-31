#!/usr/bin/env python3
"""
Blender version of render_paired_videos_simple.py
Generates paired videos (with/without human) and mask using Blender's rendering engine.

Usage:
    blender --background --python render_paired_videos_blender.py -- [args]
"""

import os
import sys
import argparse
import random
import json
import numpy as np
import math
import pickle

# Add script directory to path to import blender_utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Ensure Blender's Python can find pip-installed packages (e.g., opencv)
import site
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

# Physics configuration file
PHYSICS_CONFIG_FILE = os.path.join(script_dir, "physics_config.json")

# Import texture mapping module
import object_texture_mapping

# Import Blender Python API
try:
    import bpy
    import mathutils
except ImportError:
    print("ERROR: This script must be run with Blender's Python interpreter")
    print("Usage: blender --background --python render_paired_videos_blender.py -- [args]")
    sys.exit(1)

# Import our utilities
try:
    from blender_utils import (
        load_humoto_pickle, extract_pose_params, load_object_model,
        subsample_sequence, get_light_colors, extract_hip_position,
        calculate_camera_from_hip, get_random_camera_motion, animate_camera as animate_camera_motion
    )
except ImportError as e:
    print(f"ERROR: Could not import blender_utils: {e}")
    print(f"Make sure blender_utils.py is in {script_dir}")
    sys.exit(1)


# Coordinate system conversion: Y-up (X right, Y up, Z fwd) -> Blender (X right, Y fwd, Z up)
_R = np.array([[1, 0, 0],
               [0, 0, 1],
               [0, 1, 0]], dtype=float)   # maps (x,y,z)->(x,z,y)


def rotmat_yup_to_blender(Rsrc3x3):
    """Convert 3x3 rotation matrix from Y-up to Blender coordinate system."""
    Rsrc = np.asarray(Rsrc3x3, dtype=float).reshape(3, 3)
    return _R @ Rsrc @ _R.T


def quat_yup_to_blender(qwxyz):
    """Convert quaternion from Y-up to Blender coordinate system.

    Args:
        qwxyz: Quaternion in [w, x, y, z] format

    Returns:
        mathutils.Quaternion in Blender coordinate system
    """
    q = mathutils.Quaternion((float(qwxyz[0]),
                              float(qwxyz[1]),
                              float(qwxyz[2]),
                              float(qwxyz[3])))
    Rb = rotmat_yup_to_blender(np.array(q.to_matrix()))
    return mathutils.Matrix(Rb).to_quaternion()


def load_physics_config(sequence_name):
    """Load physics configuration for a specific sequence.

    Returns:
        dict: Configuration with 'static_objects' and 'physics_objects' lists,
              or None if no config exists for this sequence.
    """
    if not os.path.exists(PHYSICS_CONFIG_FILE):
        return None

    try:
        with open(PHYSICS_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get(sequence_name)
    except Exception as e:
        print(f"Warning: Could not load physics config: {e}")
        return None


def parse_args():
    """Parse command line arguments after '--' separator."""
    # Blender adds its own args before '--', we only want args after it
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        # Running directly with Python (not through Blender), use all args
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Generate paired videos with Blender')
    parser.add_argument("-d", "--dataset_dir", type=str, required=True,
                        help="Path to the HUMOTO dataset directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory for the paired videos")
    parser.add_argument("-s", "--sequences", type=str, nargs='+', required=True,
                        help="Specific sequence names to process")
    parser.add_argument("-m", "--object_model", type=str, required=True,
                        help="Path to the object model directory")
    parser.add_argument("-y", "--y_up", action='store_true', default=True,
                        help="Whether to render in y-up coordinate system")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for camera trajectory")
    parser.add_argument("--fps", type=int, default=12,
                        help="Frames per second for output videos")
    parser.add_argument("--target_frames", type=int, default=60,
                        help="Target number of frames (default: 60 for 5 sec at 12fps)")
    parser.add_argument("--resolution_x", type=int, default=1600,
                        help="Render resolution X")
    parser.add_argument("--resolution_y", type=int, default=900,
                        help="Render resolution Y")
    parser.add_argument("--floor_texture", type=str, default=None,
                        help="Path to floor texture directory or 'random' to select random texture")
    parser.add_argument("--wall_texture", type=str, default=None,
                        help="Path to wall texture directory or 'random' to select random texture")
    parser.add_argument("--object_texture", type=str, default=None,
                        help="Path to object texture directory or 'random' to select random texture")
    parser.add_argument("--add_walls", action='store_true', default=False,
                        help="Add walls to the scene")
    parser.add_argument("--enable_physics", action='store_true', default=True,
                        help="Enable physics simulation (objects fall when human removed)")
    parser.add_argument("--random_human_color", action='store_true', default=False,
                        help="Apply random solid color to human instead of default grey")
    parser.add_argument("--rename_output", type=str, default=None,
                        help="Rename output files to this name instead of using sequence name")
    parser.add_argument("--use_characters", action='store_true', default=False,
                        help="Use textured characters (Remy/Sophie) instead of solid color human")
    parser.add_argument("--characters_dir", type=str, default="./humoto_characters_converted",
                        help="Path to converted character datasets directory")
    parser.add_argument("--force_character", type=str, default=None, choices=['remy', 'sophie'],
                        help="Force specific character instead of random selection")
    parser.add_argument("--camera_variant", type=str, default='v1',
                        choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9'],
                        help="Camera diversity level (v1-v3: standard, v4-v6: increased movement/zoom)")

    return parser.parse_args(argv)


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear all meshes, materials, etc.
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)


def setup_scene(resolution_x=1600, resolution_y=900, fps=12):
    """Setup basic Blender scene for rendering."""
    scene = bpy.context.scene

    # Set render settings
    # BLENDER_EEVEE was renamed to BLENDER_EEVEE_NEXT in Blender 4.2+
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.fps = fps

    # Set output format
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'

    # Enable transparent background for mask rendering
    scene.render.film_transparent = False

    # Setup world background (white)
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (1, 1, 1, 1)  # White
        bg_node.inputs['Strength'].default_value = 1.0

    return scene


def select_random_floor_texture(textures_base_dir='./textures/floor/organized'):
    """Select a random floor texture from available textures.

    Searches both original textures and new Blender textures for maximum variety.

    Args:
        textures_base_dir: Base directory containing texture subdirectories

    Returns:
        Path to selected texture directory, or None if not found
    """
    all_textures = []

    # Search original texture directory
    if os.path.exists(textures_base_dir):
        texture_dirs = [os.path.join(textures_base_dir, d)
                       for d in os.listdir(textures_base_dir)
                       if os.path.isdir(os.path.join(textures_base_dir, d))]
        all_textures.extend(texture_dirs)

    # Search new Blender textures directory
    blender_floors = './Blender/floors-bl'
    if os.path.exists(blender_floors):
        blender_dirs = [os.path.join(blender_floors, d)
                       for d in os.listdir(blender_floors)
                       if os.path.isdir(os.path.join(blender_floors, d))]
        all_textures.extend(blender_dirs)

    if not all_textures:
        print(f"WARNING: No floor textures found in {textures_base_dir} or {blender_floors}")
        return None

    # Select random texture from combined pool
    texture_path = random.choice(all_textures)
    selected_name = os.path.basename(texture_path)
    print(f"Selected random floor texture: {selected_name} (from pool of {len(all_textures)})")
    return texture_path


def create_ground_plane(size=10, y_up=True, texture_path=None):
    """Create a ground plane with optional PBR textures or chessboard pattern.

    Args:
        size: Size of the plane
        y_up: Whether using Y-up coordinate system
        texture_path: Path to texture directory (should contain Color, Normal, Roughness, Displacement)
                     If None, creates chessboard pattern
    """
    import math

    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    # After transforming data to Z-up, ground should be in XY plane (Blender default)
    # No rotation needed

    # Create material
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    ground.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create basic shader nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    coord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')

    # Configure mapping for tiling
    mapping.inputs['Scale'].default_value = (size/2, size/2, 1)  # Tile the texture

    if texture_path and os.path.exists(texture_path):
        # Load PBR textures
        print(f"Loading floor textures from: {texture_path}")

        # Find texture files
        texture_files = os.listdir(texture_path)
        color_tex = None
        normal_tex = None
        roughness_tex = None
        displacement_tex = None

        for f in texture_files:
            full_path = os.path.join(texture_path, f)
            f_lower = f.lower()
            # Support both original naming (Color, NormalGL) and Blender naming (albedo, normal-ogl)
            if 'color' in f_lower or 'albedo' in f_lower:
                color_tex = full_path
            elif 'normalgl' in f_lower or 'normal-ogl' in f_lower or 'normal_ogl' in f_lower:
                normal_tex = full_path
            elif 'roughness' in f_lower:
                roughness_tex = full_path
            elif 'displacement' in f_lower or 'height' in f_lower:
                displacement_tex = full_path

        # Load color texture
        if color_tex:
            img_color = bpy.data.images.load(color_tex)
            tex_color = nodes.new('ShaderNodeTexImage')
            tex_color.image = img_color
            links.new(mapping.outputs['Vector'], tex_color.inputs['Vector'])
            links.new(tex_color.outputs['Color'], bsdf.inputs['Base Color'])

        # Load normal map
        if normal_tex:
            img_normal = bpy.data.images.load(normal_tex)
            img_normal.colorspace_settings.name = 'Non-Color'
            tex_normal = nodes.new('ShaderNodeTexImage')
            tex_normal.image = img_normal
            normal_map = nodes.new('ShaderNodeNormalMap')
            links.new(mapping.outputs['Vector'], tex_normal.inputs['Vector'])
            links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

        # Load roughness
        if roughness_tex:
            img_roughness = bpy.data.images.load(roughness_tex)
            img_roughness.colorspace_settings.name = 'Non-Color'
            tex_roughness = nodes.new('ShaderNodeTexImage')
            tex_roughness.image = img_roughness
            links.new(mapping.outputs['Vector'], tex_roughness.inputs['Vector'])
            links.new(tex_roughness.outputs['Color'], bsdf.inputs['Roughness'])

        print(f"Loaded floor textures: Color={color_tex is not None}, Normal={normal_tex is not None}, Roughness={roughness_tex is not None}")

    else:
        # Fallback to chessboard pattern
        checker = nodes.new('ShaderNodeTexChecker')
        checker.inputs['Scale'].default_value = size * 2
        checker.inputs['Color1'].default_value = (0.8, 0.8, 0.8, 1)  # Light grey
        checker.inputs['Color2'].default_value = (0.2, 0.2, 0.2, 1)  # Dark grey
        links.new(mapping.outputs['Vector'], checker.inputs['Vector'])
        links.new(checker.outputs['Color'], bsdf.inputs['Base Color'])

    # Link coordinate system
    links.new(coord.outputs['UV'], mapping.inputs['Vector'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return ground


def select_random_wall_texture(textures_base_dir='./textures/wall/organized'):
    """Select a random wall texture from available textures.

    Searches both original textures and new Blender textures for maximum variety.
    """
    all_textures = []

    # Search original texture directory
    if os.path.exists(textures_base_dir):
        texture_dirs = [os.path.join(textures_base_dir, d)
                       for d in os.listdir(textures_base_dir)
                       if os.path.isdir(os.path.join(textures_base_dir, d))]
        all_textures.extend(texture_dirs)

    # Search new Blender textures directory
    blender_walls = './Blender/walls-bl'
    if os.path.exists(blender_walls):
        blender_dirs = [os.path.join(blender_walls, d)
                       for d in os.listdir(blender_walls)
                       if os.path.isdir(os.path.join(blender_walls, d))]
        all_textures.extend(blender_dirs)

    if not all_textures:
        print(f"WARNING: No wall textures found in {textures_base_dir} or {blender_walls}")
        return None

    # Select random texture from combined pool
    texture_path = random.choice(all_textures)
    selected_name = os.path.basename(texture_path)
    print(f"Selected random wall texture: {selected_name} (from pool of {len(all_textures)})")
    return texture_path


def select_random_object_texture(textures_base_dir='./textures/general/organized'):
    """Select a random object texture from available textures.

    Searches both original textures and new Blender textures for maximum variety.
    Includes wood, metal, countertops, fabric, and synthetic textures.
    """
    all_textures = []

    # Search original texture directory
    if os.path.exists(textures_base_dir):
        texture_dirs = [os.path.join(textures_base_dir, d)
                       for d in os.listdir(textures_base_dir)
                       if os.path.isdir(os.path.join(textures_base_dir, d))]
        all_textures.extend(texture_dirs)

    # Search new Blender texture directories (multiple categories for objects)
    blender_categories = ['wood-bl', 'metals-bl', 'countertops-bl', 'fabric-bl', 'synthetic-bl']
    for category in blender_categories:
        blender_path = f'./Blender/{category}'
        if os.path.exists(blender_path):
            blender_dirs = [os.path.join(blender_path, d)
                           for d in os.listdir(blender_path)
                           if os.path.isdir(os.path.join(blender_path, d))]
            all_textures.extend(blender_dirs)

    if not all_textures:
        print(f"WARNING: No object textures found in {textures_base_dir} or Blender categories")
        return None

    # Select random texture from combined pool
    texture_path = random.choice(all_textures)
    selected_name = os.path.basename(texture_path)
    print(f"Selected random object texture: {selected_name} (from pool of {len(all_textures)})")
    return texture_path


def select_random_object_texture_unique(used_textures, textures_base_dir='./textures/general/organized'):
    """Select a random object texture that hasn't been used yet.

    Searches both original textures and new Blender textures for maximum variety.

    Args:
        used_textures: List of texture paths already used
        textures_base_dir: Base directory containing texture subdirectories

    Returns:
        Path to selected texture directory, or None if not found
    """
    all_textures = []

    # Search original texture directory
    if os.path.exists(textures_base_dir):
        texture_dirs = [os.path.join(textures_base_dir, d)
                       for d in os.listdir(textures_base_dir)
                       if os.path.isdir(os.path.join(textures_base_dir, d))]
        all_textures.extend(texture_dirs)

    # Search new Blender texture directories (multiple categories for objects)
    blender_categories = ['wood-bl', 'metals-bl', 'countertops-bl', 'fabric-bl', 'synthetic-bl']
    for category in blender_categories:
        blender_path = f'./Blender/{category}'
        if os.path.exists(blender_path):
            blender_dirs = [os.path.join(blender_path, d)
                           for d in os.listdir(blender_path)
                           if os.path.isdir(os.path.join(blender_path, d))]
            all_textures.extend(blender_dirs)

    if not all_textures:
        print(f"WARNING: No object textures found")
        return None

    # Filter out already used textures
    available_textures = [t for t in all_textures if t not in used_textures]

    # If all textures have been used, allow reuse
    if not available_textures:
        print(f"  Note: All {len(all_textures)} textures used, allowing reuse")
        available_textures = all_textures

    texture_path = random.choice(available_textures)
    selected_name = os.path.basename(texture_path)
    print(f"  Selected unique object texture: {selected_name} ({len(available_textures)} available)")
    return texture_path


def create_walls(size=10, height=5, y_up=True, texture_path=None):
    """Create 4 walls around the scene with optional PBR textures.

    Args:
        size: Size of the room (walls will be at +/- size/2)
        height: Height of the walls
        y_up: Whether using Y-up coordinate system
        texture_path: Path to texture directory for walls

    Returns:
        List of wall objects
    """
    walls = []
    half_size = size / 2

    # Make walls much larger to ensure visibility
    wall_size = size * 3  # 3x larger
    wall_distance = size  # Place walls further back

    # Wall configurations (in Blender Z-up coordinates)
    # Each wall is a vertical plane facing inward
    wall_configs = [
        {'name': 'WallBack', 'location': (0, -wall_distance, height/2), 'rotation': (math.pi/2, 0, 0)},  # Behind camera
        {'name': 'WallLeft', 'location': (-wall_distance, 0, height/2), 'rotation': (math.pi/2, 0, math.pi/2)},  # Left side
        {'name': 'WallRight', 'location': (wall_distance, 0, height/2), 'rotation': (math.pi/2, 0, -math.pi/2)},  # Right side
    ]

    for config in wall_configs:
        # Create plane for wall
        bpy.ops.mesh.primitive_plane_add(
            size=wall_size,
            location=config['location'],
            rotation=config['rotation']
        )
        wall = bpy.context.active_object
        wall.name = config['name']

        # Scale height
        wall.scale[2] = height / wall_size  # Adjust height scaling

        # Apply texture if provided
        if texture_path and os.path.exists(texture_path):
            apply_pbr_texture(wall, texture_path, scale=(size/2, height/2, 1))
        else:
            # Default grey material
            mat = bpy.data.materials.new(name=f"{config['name']}_Material")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.9, 1)
            wall.data.materials.append(mat)

        walls.append(wall)

    return walls


def apply_pbr_texture(obj, texture_path, scale=(1, 1, 1)):
    """Apply PBR textures to an object.

    Args:
        obj: Blender object to apply texture to
        texture_path: Path to texture directory
        scale: Texture scale (x, y, z)
    """
    # Create material
    mat = bpy.data.materials.new(name=f"{obj.name}_Material")
    mat.use_nodes = True
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create basic shader nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    coord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')

    # Configure mapping for tiling
    mapping.inputs['Scale'].default_value = scale

    # Find texture files
    texture_files = os.listdir(texture_path)
    color_tex = None
    normal_tex = None
    roughness_tex = None

    for f in texture_files:
        full_path = os.path.join(texture_path, f)
        f_lower = f.lower()
        # Support multiple naming conventions: Color/albedo, NormalGL/normal-ogl, etc.
        if 'color' in f_lower or 'albedo' in f_lower or 'basecolor' in f_lower:
            color_tex = full_path
        elif 'normalgl' in f_lower or 'normal-ogl' in f_lower or 'normal_ogl' in f_lower or ('normal' in f_lower and 'ogl' in f_lower):
            normal_tex = full_path
        elif 'roughness' in f_lower:
            roughness_tex = full_path

    # Load color texture
    if color_tex:
        img_color = bpy.data.images.load(color_tex)
        tex_color = nodes.new('ShaderNodeTexImage')
        tex_color.image = img_color
        links.new(mapping.outputs['Vector'], tex_color.inputs['Vector'])
        links.new(tex_color.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        # Fallback: Set a default mid-gray color if no color texture found
        print(f"  Warning: No color texture found in {texture_path}, using fallback gray")
        bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)

    # Load normal map
    if normal_tex:
        img_normal = bpy.data.images.load(normal_tex)
        img_normal.colorspace_settings.name = 'Non-Color'
        tex_normal = nodes.new('ShaderNodeTexImage')
        tex_normal.image = img_normal
        normal_map = nodes.new('ShaderNodeNormalMap')
        links.new(mapping.outputs['Vector'], tex_normal.inputs['Vector'])
        links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

    # Load roughness
    if roughness_tex:
        img_roughness = bpy.data.images.load(roughness_tex)
        img_roughness.colorspace_settings.name = 'Non-Color'
        tex_roughness = nodes.new('ShaderNodeTexImage')
        tex_roughness.image = img_roughness
        links.new(mapping.outputs['Vector'], tex_roughness.inputs['Vector'])
        links.new(tex_roughness.outputs['Color'], bsdf.inputs['Roughness'])

    # Link coordinate system (use Generated for better texture mapping on procedural geometry)
    links.new(coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])


def setup_lighting(y_up=True, seed=None):
    """Setup scene lighting with optional randomization.

    Args:
        y_up: Whether using Y-up coordinate system
        seed: Random seed for lighting variation
    """
    import numpy as np

    # Add variability based on seed
    if seed is not None:
        np.random.seed(seed + 1)  # +1 to differ from camera seed

    # Randomize sun light position and energy
    sun_x = np.random.uniform(-8, 8)  # Vary horizontal position
    sun_y = np.random.uniform(3, 8)   # Vary height (keep elevated)
    sun_z = np.random.uniform(5, 12)  # Vary depth
    sun_energy = np.random.uniform(0.3, 0.7)  # Vary brightness

    bpy.ops.object.light_add(type='SUN', location=(sun_x, sun_y, sun_z))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = sun_energy

    # Randomize fill light energy and size
    fill_energy = np.random.uniform(5.0, 12.0)  # Vary ambient light
    fill_size = np.random.uniform(8, 12)         # Vary light spread

    # Add ambient/fill light
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    area = bpy.context.active_object
    area.name = "FillLight"
    area.data.energy = fill_energy
    area.data.size = fill_size

    print(f"  Lighting randomized: Sun(pos=[{sun_x:.1f}, {sun_y:.1f}, {sun_z:.1f}], energy={sun_energy:.2f}), Fill(energy={fill_energy:.1f}, size={fill_size:.1f})")

    return sun, area




def create_human_mesh(vertices, faces, name="Human", y_up=True, color=None):
    """
    Create human mesh from pre-computed vertices and faces.

    Args:
        vertices: numpy array [num_verts, 3] - first frame vertices
        faces: numpy array [num_faces, 3] - face indices
        name: name for the mesh object
        y_up: whether input data is in Y-up system
        color: optional RGB tuple (r, g, b) for human color, defaults to grey

    Returns:
        Blender object with the human mesh
    """
    # Transform from Y-up to Z-up (Blender) coordinate system
    # Input (from HumanModel with y_up=True): X=right, Y=up, Z=forward
    # Blender: X=right, Y=forward, Z=up
    # Transformation: (x, y, z)_yup → (x, z, y)_blender
    if y_up:
        transformed = np.zeros_like(vertices)
        transformed[:, 0] = vertices[:, 0]    # X stays same (right)
        transformed[:, 1] = vertices[:, 2]    # Blender Y (forward) = Y-up Z (forward)
        transformed[:, 2] = vertices[:, 1]    # Blender Z (up) = Y-up Y (up)
        vertices = transformed

    # Convert numpy arrays to lists for Blender
    verts_list = vertices.tolist()
    faces_list = faces.tolist()

    # Create mesh
    mesh = bpy.data.meshes.new(name=f"{name}_Mesh")
    mesh.from_pydata(verts_list, [], faces_list)
    mesh.update()

    # Create object
    human_obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(human_obj)

    # Create material with specified color or default grey
    if color is None:
        color = (0.7, 0.7, 0.7)  # Default grey

    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (*color, 1)
        bsdf.inputs['Roughness'].default_value = 0.95  # More matte to prevent washout
        # Blender 4.0+ changed Specular to "Specular IOR Level"
        if 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = 0.0
        elif 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = 0.0
    human_obj.data.materials.append(mat)

    return human_obj


def animate_human_mesh(human_obj, all_vertices, y_up=True):
    """
    Animate human mesh by creating shape keys for each frame.

    Args:
        human_obj: Blender object with the human mesh
        all_vertices: numpy array [num_frames, num_verts, 3]
        y_up: whether input data is in Y-up system (will be converted to Z-up)
    """
    num_frames = len(all_vertices)

    # Transform from Y-up to Z-up (Blender): (x, y, z) → (x, z, y)
    if y_up:
        transformed = np.zeros_like(all_vertices)
        transformed[:, :, 0] = all_vertices[:, :, 0]   # X stays same
        transformed[:, :, 1] = all_vertices[:, :, 2]   # Blender Y = Y-up Z
        transformed[:, :, 2] = all_vertices[:, :, 1]   # Blender Z = Y-up Y
        all_vertices = transformed

    # Add basis shape key (reference)
    if not human_obj.data.shape_keys:
        basis = human_obj.shape_key_add(name='Basis')
        basis.interpolation = 'KEY_LINEAR'

    # Create a shape key for each frame
    shape_keys = []
    for frame_idx in range(num_frames):
        sk = human_obj.shape_key_add(name=f'Frame_{frame_idx:04d}')
        sk.interpolation = 'KEY_LINEAR'

        # Update shape key vertex positions
        frame_verts = all_vertices[frame_idx]
        for vert_idx, vert_co in enumerate(frame_verts):
            sk.data[vert_idx].co = vert_co

        shape_keys.append(sk)

    # Animate shape key values
    # Only one shape key should be at value 1.0 per frame
    for frame_idx in range(num_frames):
        for sk_idx, sk in enumerate(shape_keys):
            if sk_idx == frame_idx:
                sk.value = 1.0
            else:
                sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=frame_idx + 1)

    print(f"Animated human mesh with {num_frames} shape keys")


def create_armature_from_bones(bone_data, name="Armature"):
    """Create Blender armature from bone hierarchy."""
    # Create armature
    armature = bpy.data.armatures.new(name=f"{name}_Armature")
    armature_obj = bpy.data.objects.new(name, armature)
    bpy.context.collection.objects.link(armature_obj)

    # Enter edit mode to create bones
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Create bones
    edit_bones = armature.edit_bones
    bone_map = {}

    for bone_name, bone_info in bone_data.items():
        bone = edit_bones.new(bone_name)
        # Set bone positions (these will be updated during animation)
        bone.head = (0, 0, 0)
        bone.tail = (0, 0.1, 0)  # Small offset
        bone_map[bone_name] = bone

    bpy.ops.object.mode_set(mode='OBJECT')

    return armature_obj, bone_map


def load_object_mesh(obj_path):
    """Load object mesh from file."""
    if obj_path.endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=obj_path)
    elif obj_path.endswith('.ply'):
        bpy.ops.import_mesh.ply(filepath=obj_path)
    elif obj_path.endswith('.stl'):
        bpy.ops.import_mesh.stl(filepath=obj_path)
    else:
        print(f"Warning: Unsupported object format: {obj_path}")
        return None

    # Get imported object
    obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
    return obj


def create_object_from_vertices(vertices, faces, name, color=None):
    """Create Blender object from vertices and faces."""
    mesh = bpy.data.meshes.new(name=f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Create material with color
    if color is not None:
        mat = bpy.data.materials.new(name=f"{name}_Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (*color, 1)
            bsdf.inputs['Roughness'].default_value = 0.5
        obj.data.materials.append(mat)

    return obj


def setup_camera(location=(5, -5, 3), look_at=(0, 0, 1), y_up=True):
    """Setup camera."""
    # Transform from Y-up to Z-up (Blender): (x, y, z) → (x, z, y)
    if y_up:
        location = (location[0], location[2], location[1])
        look_at = (look_at[0], look_at[2], look_at[1])

    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.active_object
    camera.name = "Camera"

    # Use track-to constraint for stable, level camera
    # This ensures camera stays level and points at target
    constraint = camera.constraints.new(type='TRACK_TO')

    # Create empty at look_at position as target
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=look_at)
    target = bpy.context.active_object
    target.name = "CameraTarget"

    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'  # Camera's local up axis in Blender is Y

    # Set camera as active
    bpy.context.scene.camera = camera

    # Set camera properties
    camera.data.lens = 50  # Focal length in mm

    return camera


def generate_camera_trajectory(base_location, base_rotation, num_frames, seed=None, y_up=True):
    """Generate smooth linear camera movement (left or right)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Randomly choose direction: left (-1) or right (+1)
    direction = random.choice([-1, 1])

    # Total movement distance (in Blender units)
    total_movement = np.random.uniform(1.5, 3.0)

    # Generate linear trajectory
    t = np.linspace(0, 1, num_frames)  # 0 to 1 for linear interpolation
    trajectory = []

    for i in range(num_frames):
        # Linear movement along X axis (left/right in Blender)
        delta_x = direction * total_movement * t[i]

        # Base location is already transformed to Z-up at this point
        location = (
            base_location[0] + delta_x,
            base_location[1],  # No Y movement
            base_location[2]   # No Z movement (height stays constant)
        )

        trajectory.append({
            'location': location,
            'rotation': base_rotation  # Keep rotation fixed for now
        })

    return trajectory


def animate_camera(camera, trajectory):
    """Animate camera along trajectory."""
    for frame, pose in enumerate(trajectory, start=1):
        camera.location = pose['location']
        camera.keyframe_insert(data_path="location", frame=frame)


def update_human_pose(human_obj, pose_params, frame):
    """Update human mesh vertices for given frame and pose parameters."""
    # This is a simplified version - in practice you'd apply bone transforms
    # to deform the mesh based on the armature
    # For now, we'll just apply a simple transform

    # TODO: Implement proper skinning with armature
    pass


def animate_objects(objects_data, object_pose_params, num_frames, y_up=True):
    """Animate object transformations."""
    for obj_name, obj_info in objects_data.items():
        obj = obj_info['object']

        if obj_name not in object_pose_params:
            print(f"  Warning: No pose data for {obj_name}")
            continue

        poses = object_pose_params[obj_name]  # Array of [w, x, y, z, tx, ty, tz]

        # Animate each frame
        for frame in range(min(num_frames, len(poses))):
            pose = poses[frame]

            # Extract rotation (quaternion) and translation
            # Pose format: [w, x, y, z, tx, ty, tz]
            w, x, y, z = float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])
            tx, ty, tz = float(pose[4]), float(pose[5]), float(pose[6])

            # Transform position from Y-up to Z-up (Blender): (x, y, z) → (x, z, y)
            if y_up:
                tx, ty, tz = tx, tz, ty

            obj.location = (tx, ty, tz)
            obj.keyframe_insert(data_path="location", frame=frame + 1)

            # Transform quaternion from Y-up to Z-up (Blender) using rotation matrix sandwich
            if y_up:
                q_bl = quat_yup_to_blender((w, x, y, z))
            else:
                q_bl = mathutils.Quaternion((w, x, y, z))

            # Use quaternion rotation mode to avoid Euler gimbal issues
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = q_bl
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame + 1)


def render_video(output_path, num_frames, scene):
    """Render animation to video."""
    scene.frame_start = 1
    scene.frame_end = num_frames

    # Set output path
    scene.render.filepath = output_path

    # Render animation
    # Note: Blender doesn't directly render to video in background mode easily
    # So we render to image sequence and then combine with ffmpeg

    temp_dir = output_path + "_frames"
    os.makedirs(temp_dir, exist_ok=True)

    scene.render.filepath = os.path.join(temp_dir, "frame_")

    # Render frames
    for frame in range(1, num_frames + 1):
        scene.frame_set(frame)
        scene.render.filepath = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)

    return temp_dir


def frames_to_video(frames_dir, output_path, fps=12, is_mask=False):
    """Convert image sequence to video using ffmpeg.

    Args:
        frames_dir: Directory containing frame_%04d.png files
        output_path: Output video path
        fps: Frames per second
        is_mask: If True, use lossless RGB encoding to preserve exact pixel values
    """
    import subprocess

    if is_mask:
        # Use near-lossless grayscale encoding for masks
        # yuv420p ensures broad player compatibility; crf=0 preserves discrete values
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-crf', '0',
            '-pix_fmt', 'yuv420p',
            '-preset', 'veryslow',
            output_path
        ]
    else:
        # Standard lossy H.264 encoding for RGB videos
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            output_path
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        return False


def create_trimask_from_frames(frames_with_human, frames_background_original, frames_without_human, frames_human_mask,
                               output_dir, threshold=8, human_dilate_px=3, change_erode_px=1):
    """
    Build quad-mask using 4 renders:
      0   (black)      : human only (no scene change underneath)
      63  (dark gray)  : overlap (human covering scene change)
      127 (gray)       : scene changed due to removal (shadows, reflections, physics)
      255 (white)      : unchanged

    Args:
        frames_with_human: Directory with frames including human
        frames_background_original: Directory with frames of background without human (original object positions)
        frames_without_human: Directory with frames without human (after physics/removal)
        frames_human_mask: Directory with human-only mask frames
        output_dir: Output directory for mask frames
        threshold: Pixel difference threshold for detecting scene changes (default 8)
        human_dilate_px: Pixels to dilate human mask to kill gray halos (default 3)
        change_erode_px: Pixels to erode change mask to remove noise (default 1)

    Returns:
        Output directory path
    """
    import cv2
    os.makedirs(output_dir, exist_ok=True)

    def gamma_approx(img_bgr_uint8, gamma=2.2):
        """Convert to float [0,1], simple inverse gamma (approx linearize)"""
        x = img_bgr_uint8.astype(np.float32) / 255.0
        x = np.power(np.clip(x, 1e-6, 1.0), gamma)  # cheap & cheerful
        return x

    frames = sorted([f for f in os.listdir(frames_with_human) if f.endswith('.png')])
    print(f"[create_trimask] Processing {len(frames)} frames")
    print(f"[create_trimask] Paths:")
    print(f"  with_human={frames_with_human}")
    print(f"  bg_orig={frames_background_original}")
    print(f"  without_human={frames_without_human}")
    print(f"  human_mask={frames_human_mask}")

    for fn in frames:
        p_with   = os.path.join(frames_with_human, fn)
        p_bg_orig = os.path.join(frames_background_original, fn)
        p_wo     = os.path.join(frames_without_human, fn)
        p_h_mask = os.path.join(frames_human_mask, fn)
        if not (os.path.exists(p_with) and os.path.exists(p_bg_orig) and os.path.exists(p_wo) and os.path.exists(p_h_mask)):
            print(f"[create_trimask] WARNING: Missing file for {fn}")
            continue

        img_w  = cv2.imread(p_with,   cv2.IMREAD_COLOR)
        img_bg_orig = cv2.imread(p_bg_orig, cv2.IMREAD_COLOR)
        img_wo = cv2.imread(p_wo,     cv2.IMREAD_COLOR)
        # Read human mask as RGB and convert to grayscale (not alpha channel)
        hmask_rgb = cv2.imread(p_h_mask, cv2.IMREAD_COLOR)

        if img_w is None or img_bg_orig is None or img_wo is None or hmask_rgb is None:
            print(f"[create_trimask] WARNING: Failed to read images for {fn}")
            continue

        hmask_gray = cv2.cvtColor(hmask_rgb, cv2.COLOR_BGR2GRAY)

        # --- 1) Solid human mask (kill alpha-edge fringes)
        _, human_bin = cv2.threshold(hmask_gray, 1, 255, cv2.THRESH_BINARY)

        # Debug: print stats for first frame
        if fn == frames[0]:
            print(f"[create_trimask] First frame stats:")
            print(f"  hmask_rgb shape: {hmask_rgb.shape}")
            print(f"  hmask_rgb range: [{hmask_rgb.min()}, {hmask_rgb.max()}]")
            print(f"  hmask_gray range: [{hmask_gray.min()}, {hmask_gray.max()}]")
            print(f"  hmask_gray unique values: {np.unique(hmask_gray)[:20]}")  # Show first 20 unique values
            print(f"  human_bin threshold result: min={human_bin.min()}, max={human_bin.max()}")
            print(f"  human_bin pixels > 0: {(human_bin > 0).sum()} / {human_bin.size} ({100*(human_bin > 0).sum()/human_bin.size:.1f}%)")
            print(f"  img_w shape: {img_w.shape}, img_wo shape: {img_wo.shape}")
        if human_dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*human_dilate_px+1, 2*human_dilate_px+1))
            human_bin = cv2.dilate(human_bin, k)

        # --- 2) Robust diff (approx linear domain + max-channel distance)
        # IMPORTANT: Use background_original (without human) vs rgb_removed (without human)
        # This gives us the TRUE scene change (not including the human itself)
        lbg_orig = gamma_approx(img_bg_orig)
        lwo = gamma_approx(img_wo)
        diff = np.abs(lbg_orig - lwo)               # HxWx3 in [0,1]
        diff_scalar = (diff.max(axis=2) * 255.0).astype(np.uint8)  # HxW in [0,255]

        # --- 3) Decide changed areas, but never over human
        # NOTE: threshold ~8–12 usually works; you set default=8
        change = (diff_scalar >= threshold).astype(np.uint8) * 255

        # Optional small cleanup of change mask (remove speckles / thin lines)
        if change_erode_px > 0:
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*change_erode_px+1, 2*change_erode_px+1))
            change = cv2.morphologyEx(change, cv2.MORPH_OPEN, k2)

        if fn == frames[0]:
            print(f"  Human pixels: {(human_bin > 0).sum():,} ({(human_bin > 0).sum()/human_bin.size*100:.2f}%)")
            print(f"  Change pixels: {(change > 0).sum():,} ({(change > 0).sum()/change.size*100:.2f}%)")
            print(f"  Diff_scalar stats: min={diff_scalar.min()}, max={diff_scalar.max()}, mean={diff_scalar.mean():.1f}")

        # --- 4) Detect overlap BEFORE excluding human from change
        # Overlap = where BOTH human AND scene change exist
        overlap = (change > 0) & (human_bin > 0)

        # Pure human = human with NO scene change underneath
        pure_human = (human_bin > 0) & (change == 0)

        # Pure change = scene change NOT under human
        pure_change = (change > 0) & (human_bin == 0)

        # --- 5) Create 3 separate masks

        # Mask 1: Human binary mask (0 = human, 255 = background)
        human_mask = np.full_like(diff_scalar, 255, dtype=np.uint8)
        human_mask[human_bin > 0] = 0

        # Mask 2: Change binary mask (127 = change NOT under human, 255 = background)
        change_mask = np.full_like(diff_scalar, 255, dtype=np.uint8)
        change_mask[pure_change] = 127

        # Mask 3: Quad-mask (0, 63, 127, 255)
        quad_mask = np.full_like(diff_scalar, 255, dtype=np.uint8)
        quad_mask[pure_change] = 127       # Grey: scene change only
        quad_mask[pure_human] = 0           # Black: human only
        quad_mask[overlap] = 63             # Dark grey: human + scene change (overlap)

        # Debug: print quad-mask stats for first frame
        if fn == frames[0]:
            black_pixels = (quad_mask == 0).sum()
            overlap_pixels = (quad_mask == 63).sum()
            gray_pixels = (quad_mask == 127).sum()
            white_pixels = (quad_mask == 255).sum()
            total_pixels = quad_mask.size
            print(f"[create_trimask] First frame quad-mask composition:")
            print(f"  Black (human only): {black_pixels} ({100*black_pixels/total_pixels:.1f}%)")
            print(f"  Dark Gray (overlap): {overlap_pixels} ({100*overlap_pixels/total_pixels:.1f}%)")
            print(f"  Gray (change only): {gray_pixels} ({100*gray_pixels/total_pixels:.1f}%)")
            print(f"  White (keep): {white_pixels} ({100*white_pixels/total_pixels:.1f}%)")

        # Save all 3 masks as 3-channel BGR (R=G=B) to preserve exact pixel values
        # Create subdirectories for each mask type
        human_mask_dir = output_dir + "_human"
        change_mask_dir = output_dir + "_change"
        quad_mask_dir = output_dir

        os.makedirs(human_mask_dir, exist_ok=True)
        os.makedirs(change_mask_dir, exist_ok=True)
        os.makedirs(quad_mask_dir, exist_ok=True)

        # Convert to BGR and save
        human_mask_bgr = cv2.cvtColor(human_mask, cv2.COLOR_GRAY2BGR)
        change_mask_bgr = cv2.cvtColor(change_mask, cv2.COLOR_GRAY2BGR)
        quad_mask_bgr = cv2.cvtColor(quad_mask, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(os.path.join(human_mask_dir, fn), human_mask_bgr)
        cv2.imwrite(os.path.join(change_mask_dir, fn), change_mask_bgr)
        cv2.imwrite(os.path.join(quad_mask_dir, fn), quad_mask_bgr)

    print(f"[create_trimask] Completed {len(frames)} frames -> {output_dir}")
    return output_dir


def load_character_fbx(fbx_path):
    """Load character FBX file (Remy or Sophie).

    Returns armature, character meshes, and scene objects dict.
    The scene objects from the FBX are at the correct scale relative to the character.
    """
    print(f"  Loading character FBX: {fbx_path}")
    # The conversion process (clear_human_scale.sh + transfer_human_model.sh)
    # already handles all coordinate transformations - import as-is
    bpy.ops.import_scene.fbx(
        filepath=fbx_path,
        axis_forward='Z',    # Match converted FBX forward direction
        axis_up='Z'          # Keep Z-up (Blender default)
    )

    armature = None
    meshes = []
    scene_objects = {}

    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            armature = obj
        elif obj.type == 'MESH':
            # The converted FBX contains both character meshes AND scene objects
            if obj.parent and obj.parent.type == 'ARMATURE':
                # This is a character mesh (Body, Bottoms, Hair, etc.)
                meshes.append(obj)
            else:
                # This is a scene object (table, mixing_bowl, etc.) - keep it!
                # It's at the correct scale relative to the character
                print(f"  Found scene object in FBX: {obj.name}")
                scene_objects[obj.name] = obj

    return armature, meshes, scene_objects


def apply_matte_materials(meshes):
    """
    Apply matte material properties to character meshes.
    Makes characters look realistic and non-shiny.
    """
    print(f"  Applying matte materials to {len(meshes)} meshes...")

    for mesh in meshes:
        if not mesh.data.materials:
            continue

        for mat in mesh.data.materials:
            if not mat or not mat.use_nodes:
                continue

            # Find Principled BSDF node
            bsdf = None
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf = node
                    break

            if not bsdf:
                continue

            # Adjust properties to reduce shine and add realism
            if 'Roughness' in bsdf.inputs:
                bsdf.inputs['Roughness'].default_value = 0.85  # More matte

            if 'Specular' in bsdf.inputs:
                bsdf.inputs['Specular'].default_value = 0.3  # Less shine
            elif 'Specular IOR Level' in bsdf.inputs:  # Blender 4.0+
                bsdf.inputs['Specular IOR Level'].default_value = 0.3

            if 'Metallic' in bsdf.inputs:
                bsdf.inputs['Metallic'].default_value = 0.0  # Non-metallic

            if 'Subsurface' in bsdf.inputs:
                bsdf.inputs['Subsurface'].default_value = 0.05  # Subtle SSS

    print(f"  ✓ Matte materials applied")


def select_character(sequence_name, force_character=None):
    """
    Select which character to use for this sequence.

    Args:
        sequence_name: Name of the sequence (for consistent random selection)
        force_character: Force specific character ('remy' or 'sophie'), or None for random

    Returns:
        str: Character name ('remy' or 'sophie')
    """
    import random

    if force_character:
        return force_character.lower()

    # Use sequence name as seed for consistent character assignment
    random.seed(hash(sequence_name))
    character = random.choice(['remy', 'sophie'])
    random.seed()  # Reset seed

    return character


def process_sequence(sequence_path, output_dir, args):
    """Process a single sequence to create paired videos."""
    import random  # Ensure random is available in function scope

    sequence_name = os.path.basename(sequence_path.rstrip('/'))

    # Use custom output name if provided
    output_name = args.rename_output if args.rename_output else sequence_name
    print(f"\nProcessing sequence: {sequence_name}")
    if args.rename_output:
        print(f"Output will be named: {output_name}")

    # Clear scene
    clear_scene()

    # Setup scene
    scene = setup_scene(
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        fps=args.fps
    )

    # Determine which directory to load sequence data from
    # If using characters, load from converted directory; otherwise load from original
    actual_sequence_path = sequence_path
    character_name = None  # Initialize to None for non-character renders
    if args.use_characters:
        # Select character first to know which converted directory to use
        character_name = select_character(sequence_name, args.force_character)
        print(f"Using character: {character_name.upper()}")

        # Path to converted character data
        # Note: conversion scripts create nested subdirectories, so we need: characters_dir/char/seq/seq/
        char_sequence_base = os.path.join(args.characters_dir, character_name, sequence_name)
        char_sequence_path = os.path.join(char_sequence_base, sequence_name)

        if not os.path.exists(char_sequence_path):
            print(f"ERROR: Converted character data not found: {char_sequence_path}")
            print(f"Please run conversion first!")
            return False

        # Use converted directory for loading sequence data
        actual_sequence_path = char_sequence_path
        print(f"Loading sequence data from converted directory: {char_sequence_path}")

    # Load sequence data using blender_utils
    try:
        data = load_humoto_pickle(actual_sequence_path, y_up=args.y_up)
        print(f"Loaded {data['num_frames']} frames with objects: {data['object_names']}")

        # IMPORTANT: Filter out character body parts from object list
        # Converted pickles incorrectly include character meshes in objects list
        # Remy meshes: Body, Bottoms, Eyes, Eyelashes, Hair, Shoes, Tops
        # Sophie meshes: Ch02_Body, Ch02_Cloth, Ch02_Eyelashes, Ch02_Hair, Ch02_Sneakers, Ch02_Socks
        character_mesh_names = {'Body', 'Bottoms', 'Eyes', 'Eyelashes', 'Hair', 'Shoes', 'Tops', 'Armature',
                                'Ch02_Body', 'Ch02_Cloth', 'Ch02_Eyelashes', 'Ch02_Hair', 'Ch02_Sneakers', 'Ch02_Socks'}
        original_objects = data['object_names'].copy()
        data['object_names'] = [obj for obj in data['object_names'] if obj not in character_mesh_names]
        if len(original_objects) != len(data['object_names']):
            filtered_out = [obj for obj in original_objects if obj not in data['object_names']]
            print(f"  Filtered out character meshes from objects: {filtered_out}")
            print(f"  Actual scene objects: {data['object_names']}")
    except Exception as e:
        print(f"Error loading sequence data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Subsample to target frames if needed
    if data['num_frames'] > args.target_frames:
        print(f"Subsampling from {data['num_frames']} to {args.target_frames} frames")
        data = subsample_sequence(data, args.target_frames)

    num_frames = data['num_frames']

    # Extract pose parameters organized by bone/object
    armature_pose_params = extract_pose_params(data['armature'])
    object_pose_params = {name: data['objects'][name] for name in data['object_names']}

    # Determine floor texture path
    floor_texture_path = None
    if args.floor_texture:
        if args.floor_texture.lower() == 'random':
            floor_texture_path = select_random_floor_texture()
        elif os.path.exists(args.floor_texture):
            floor_texture_path = args.floor_texture
            print(f"Using floor texture: {floor_texture_path}")
        else:
            print(f"WARNING: Floor texture path not found: {args.floor_texture}")

    # Create ground plane
    ground = create_ground_plane(size=10, y_up=args.y_up, texture_path=floor_texture_path)

    # Determine wall texture path
    wall_texture_path = None
    if args.add_walls and args.wall_texture:
        if args.wall_texture.lower() == 'random':
            wall_texture_path = select_random_wall_texture()
        elif os.path.exists(args.wall_texture):
            wall_texture_path = args.wall_texture
            print(f"Using wall texture: {wall_texture_path}")
        else:
            print(f"WARNING: Wall texture path not found: {args.wall_texture}")

    # Create walls if requested
    if args.add_walls:
        print("Creating walls...")
        walls = create_walls(size=10, height=5, y_up=args.y_up, texture_path=wall_texture_path)
    else:
        walls = []

    # Generate seed for randomization (lighting, camera, etc.)
    render_seed = hash(sequence_name) % 10000 if args.seed is None else args.seed

    # Setup lighting with randomization
    sun, fill_light = setup_lighting(y_up=args.y_up, seed=render_seed)

    # Load human character data
    human_obj = None
    character_meshes = []

    if args.use_characters:
        # Character already selected and path set earlier when loading sequence data
        # char_sequence_path and character_name are already defined

        print(f"Loading character FBX for {character_name.upper()}...")

        # Find character FBX file
        char_fbx = None
        for f in os.listdir(char_sequence_path):
            if f.endswith('.fbx'):
                char_fbx = os.path.join(char_sequence_path, f)
                break

        if not char_fbx or not os.path.exists(char_fbx):
            print(f"ERROR: Character FBX not found in {char_sequence_path}")
            return False

        # Load character FBX (returns armature, character meshes, and scene objects)
        armature, character_meshes, fbx_scene_objects = load_character_fbx(char_fbx)

        if not armature:
            print(f"ERROR: No armature found in {char_fbx}")
            return False

        print(f"  ✓ Loaded: {armature.name} ({len(armature.data.bones)} bones, {len(character_meshes)} meshes)")
        if fbx_scene_objects:
            print(f"  ✓ Found {len(fbx_scene_objects)} scene objects in FBX - deleting them (will load from .obj directory)")
            # Delete FBX scene objects - we'll load fresh ones from object directory
            for obj_name, obj in fbx_scene_objects.items():
                bpy.data.objects.remove(obj, do_unlink=True)
            fbx_scene_objects = {}  # Clear the dict

        # Fix: Converted FBX files consistently face backwards due to the conversion pipeline
        # The transfer_human_model.py script exports FBX with animation facing the wrong way
        # Solution: Parent armature to a Y-axis scaled empty (flips front/back without mirroring left/right)
        print(f"  Fixing character orientation by parenting to Y-scaled empty...")

        # Create empty object at armature's location
        empty = bpy.data.objects.new("CharacterOrientationFix", None)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 0.1
        bpy.context.collection.objects.link(empty)
        empty.location = armature.location.copy()

        # Scale empty by -1 on Y-axis (flips forward/backward without mirroring left/right)
        empty.scale[1] = -1.0

        # Parent armature to empty (keeps transform)
        armature.parent = empty
        armature.matrix_parent_inverse = empty.matrix_world.inverted()

        bpy.context.view_layer.update()
        print(f"  ✓ Character parented to Y-scaled empty - faces forward with correct hand movements")

        # Fix animation timing to match subsampling
        # The FBX has baked animation, but we may have subsampled the sequence
        # Need to remap FBX animation to play correctly
        if armature.animation_data and armature.animation_data.action:
            action = armature.animation_data.action
            # Get original frame count from action
            original_frames = int(action.frame_range[1])
            target_frames = num_frames  # After subsampling (e.g., 60)

            if original_frames != target_frames:
                print(f"  Remapping character animation: {original_frames} frames → {target_frames} frames")
                scale_factor = target_frames / original_frames

                # Scale all fcurves (animation channels) to fit new frame range
                for fcurve in action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        # Remap frame: old_frame * scale_factor
                        keyframe.co[0] = (keyframe.co[0] - 1) * scale_factor + 1
                        keyframe.handle_left[0] = (keyframe.handle_left[0] - 1) * scale_factor + 1
                        keyframe.handle_right[0] = (keyframe.handle_right[0] - 1) * scale_factor + 1

                # Update action frame range
                action.frame_range = (1, target_frames)
                print(f"  ✓ Character animation remapped to {target_frames} frames")

        # Apply matte materials
        apply_matte_materials(character_meshes)

        # Use first mesh as human object for hide/show
        human_obj = character_meshes[0] if character_meshes else armature

    else:
        # Use pre-computed vertices (original solid color method)
        vertices_file = os.path.join(sequence_path, f"{sequence_name}_human_verts.pkl")

        if not os.path.exists(vertices_file):
            print(f"ERROR: Pre-computed vertices not found: {vertices_file}")
            print("Please run: python precompute_human_verts.py -s {sequence_path} -o {vertices_file}")
            return False

        print(f"Loading pre-computed human vertices from {vertices_file}")
        with open(vertices_file, 'rb') as f:
            human_data = pickle.load(f)

        all_human_verts = human_data['vertices']  # [num_frames, num_verts, 3]
        human_faces = human_data['faces']          # [num_faces, 3]

        # Subsample human vertices if needed
        if len(all_human_verts) > num_frames:
            import numpy as np
            indices = np.linspace(0, len(all_human_verts) - 1, num_frames, dtype=int)
            all_human_verts = all_human_verts[indices]

    # Create/animate human (only if not using characters)
    if not args.use_characters:
        # Determine human color
        human_color = None
        if args.random_human_color:
            # Choose from dark grey, dark blue, or dark purple for better visibility
            import random
            color_choices = [
                (0.25, 0.25, 0.25, "dark grey"),      # Dark grey
                (0.15, 0.25, 0.45, "dark blue"),      # Dark blue
                (0.30, 0.15, 0.40, "dark purple")     # Dark purple
            ]
            chosen = random.choice(color_choices)
            human_color = chosen[:3]
            color_name = chosen[3]
            print(f"Using random human color: {color_name} RGB({human_color[0]:.2f}, {human_color[1]:.2f}, {human_color[2]:.2f})")

        # Create human mesh with first frame vertices
        print(f"Creating human mesh with {len(human_faces)} faces and {len(all_human_verts[0])} vertices")
        human_obj = create_human_mesh(all_human_verts[0], human_faces, name=f"Human_{sequence_name}", y_up=args.y_up, color=human_color)

        # Animate human mesh with all frames
        print("Animating human mesh...")
        animate_human_mesh(human_obj, all_human_verts, y_up=args.y_up)

    # Determine if we should use random textures for each object
    use_random_object_textures = args.object_texture and args.object_texture.lower() == 'random'
    fixed_object_texture = None
    if args.object_texture and args.object_texture.lower() != 'random':
        if os.path.exists(args.object_texture):
            fixed_object_texture = args.object_texture
            print(f"Using object texture: {fixed_object_texture}")
        else:
            print(f"WARNING: Object texture path not found: {args.object_texture}")

    # Load object meshes
    print(f"Loading {len(data['object_names'])} objects...")
    objects_data = {}
    colors = get_light_colors()

    # Track used textures to avoid repetition
    used_object_textures = []

    # IMPORTANT: Always load objects from humoto_objects directory, NOT from FBX
    # The conversion process corrupts object transforms in the FBX
    # Authors recommend: use converted FBX for character, but load objects from object directory
    # See Step 3 of conversion docs: "-m flag not recommended if you have humoto_objects_0805"
    use_fbx_objects = False

    if use_fbx_objects:
        print(f"  Using {len(fbx_scene_objects)} scene objects from character FBX (already at correct scale)")

        # Map FBX objects to the expected object names from PKL
        for obj_name in data['object_names']:
            if obj_name in fbx_scene_objects:
                obj = fbx_scene_objects[obj_name]
                print(f"  Using FBX object: {obj_name}")

                # IMPORTANT: FBX objects load with baked transforms from frame 1
                # We need to reset them to identity before applying animation from pickle
                obj.location = (0, 0, 0)
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = (1, 0, 0, 0)  # Identity quaternion
                print(f"    Reset {obj_name} to identity (was at {fbx_scene_objects[obj_name].location})")

                # Apply textures/colors to FBX objects
                color = None
                if object_texture_mapping.should_use_texture(obj_name):
                    object_texture_for_this_obj = object_texture_mapping.get_texture_for_object(obj_name)
                    if object_texture_for_this_obj and os.path.exists(object_texture_for_this_obj):
                        texture_name = os.path.basename(object_texture_for_this_obj)
                        categories = object_texture_mapping.get_texture_categories(obj_name)
                        print(f"  Applying texture to {obj_name}: {texture_name} ({', '.join(categories)})")
                        apply_pbr_texture(obj, object_texture_for_this_obj, scale=(2, 2, 2))
                    else:
                        color = object_texture_mapping.get_color_for_object(obj_name, character_name)
                        print(f"  Using color for {obj_name}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
                        mat = bpy.data.materials.new(name=f"{obj_name}_Material")
                        mat.use_nodes = True
                        bsdf = mat.node_tree.nodes.get('Principled BSDF')
                        if bsdf:
                            bsdf.inputs['Base Color'].default_value = (*color, 1)
                            bsdf.inputs['Roughness'].default_value = 0.5
                        if obj.data.materials:
                            obj.data.materials[0] = mat
                        else:
                            obj.data.materials.append(mat)
                else:
                    color = object_texture_mapping.get_color_for_object(obj_name, character_name)
                    print(f"  Applying realistic color to {obj_name}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
                    mat = bpy.data.materials.new(name=f"{obj_name}_Material")
                    mat.use_nodes = True
                    bsdf = mat.node_tree.nodes.get('Principled BSDF')
                    if bsdf:
                        bsdf.inputs['Base Color'].default_value = (*color, 1)
                        bsdf.inputs['Roughness'].default_value = 0.5
                    if obj.data.materials:
                        obj.data.materials[0] = mat
                    else:
                        obj.data.materials.append(mat)

                objects_data[obj_name] = {'object': obj, 'color': color}
            else:
                print(f"  WARNING: {obj_name} not found in FBX scene objects, skipping")
    else:
        # Original behavior: load objects from separate .obj files
        for i, obj_name in enumerate(data['object_names']):
            try:
                # Strip Blender's automatic suffix (.001, .002, etc.) for duplicate objects
                # This allows us to find the base model file
                import re
                base_obj_name = re.sub(r'\.\d+$', '', obj_name)

                # Try to load object model (using base name without suffix)
                obj_model = load_object_model(base_obj_name, args.object_model)
                print(f"  Loading object: {obj_name} from {obj_model['path']}")

                # Import the object mesh (Blender 4.0+ API)
                if obj_model['format'] == 'obj':
                    bpy.ops.wm.obj_import(filepath=obj_model['path'])
                elif obj_model['format'] == 'ply':
                    bpy.ops.wm.ply_import(filepath=obj_model['path'])
                elif obj_model['format'] == 'stl':
                    bpy.ops.wm.stl_import(filepath=obj_model['path'])
                else:
                    print(f"  Unsupported format: {obj_model['format']}, using placeholder")
                    bpy.ops.mesh.primitive_cube_add()

                # Get the imported object
                obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
                if obj:
                    obj.name = obj_name

                    # Apply rest rotation to convert Y-up authored mesh to Z-up Blender space
                    # This is a one-time transformation applied to the mesh itself before animation
                    if args.y_up:
                        obj.rotation_mode = 'XYZ'
                        obj.rotation_euler = (math.radians(90), 0, math.radians(180))  # Rotate +90° around X, then 180° around Z
                        obj.select_set(True)
                        bpy.context.view_layer.objects.active = obj
                        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
                        print(f"    Applied rotation fix: 90°X + 180°Z to {obj_name}")

                    # Apply texture or color using intelligent mapping
                    # Initialize color to None (will be set if object uses color instead of texture)
                    color = None

                    # Check if object should use texture based on material type
                    if object_texture_mapping.should_use_texture(obj_name):
                        # Get appropriate texture for this object
                        object_texture_for_this_obj = object_texture_mapping.get_texture_for_object(obj_name)

                        if object_texture_for_this_obj and os.path.exists(object_texture_for_this_obj):
                            # Apply PBR texture
                            texture_name = os.path.basename(object_texture_for_this_obj)
                            categories = object_texture_mapping.get_texture_categories(obj_name)
                            print(f"  Applying texture to {obj_name}: {texture_name} ({', '.join(categories)})")
                            apply_pbr_texture(obj, object_texture_for_this_obj, scale=(2, 2, 2))
                        else:
                            # Fallback to color if texture not found
                            color = object_texture_mapping.get_color_for_object(obj_name, character_name)
                            print(f"  Texture not found, using color for {obj_name}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
                            mat = bpy.data.materials.new(name=f"{obj_name}_Material")
                            mat.use_nodes = True
                            bsdf = mat.node_tree.nodes.get('Principled BSDF')
                            if bsdf:
                                bsdf.inputs['Base Color'].default_value = (*color, 1)
                                bsdf.inputs['Roughness'].default_value = 0.5
                            if obj.data.materials:
                                obj.data.materials[0] = mat
                            else:
                                obj.data.materials.append(mat)
                    else:
                        # Use realistic solid color for this object
                        color = object_texture_mapping.get_color_for_object(obj_name, character_name)
                        print(f"  Applying realistic color to {obj_name}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
                        mat = bpy.data.materials.new(name=f"{obj_name}_Material")
                        mat.use_nodes = True
                        bsdf = mat.node_tree.nodes.get('Principled BSDF')
                        if bsdf:
                            bsdf.inputs['Base Color'].default_value = (*color, 1)
                            bsdf.inputs['Roughness'].default_value = 0.5
                        if obj.data.materials:
                            obj.data.materials[0] = mat
                        else:
                            obj.data.materials.append(mat)

                    objects_data[obj_name] = {'object': obj, 'color': color}

            except FileNotFoundError:
                print(f"  Object model not found: {obj_name}, using placeholder")
                bpy.ops.mesh.primitive_cube_add()
                obj = bpy.context.active_object
                obj.name = obj_name
                # Use realistic color for placeholder too
                color = object_texture_mapping.get_color_for_object(obj_name, character_name)
                mat = bpy.data.materials.new(name=f"{obj_name}_Material")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get('Principled BSDF')
                if bsdf:
                    bsdf.inputs['Base Color'].default_value = (*color, 1)
                obj.data.materials.append(mat)
                objects_data[obj_name] = {'object': obj, 'color': color}

    # Calculate camera position from hip trajectory (using same seed as lighting)
    hip_positions = extract_hip_position(data['armature'])
    camera_loc, look_at = calculate_camera_from_hip(hip_positions, y_up=args.y_up, seed=render_seed,
                                                     camera_variant=args.camera_variant)

    # Setup camera
    camera = setup_camera(location=tuple(camera_loc), look_at=tuple(look_at), y_up=args.y_up)

    # Select random camera motion type and animate
    camera_motion = get_random_camera_motion(seed=render_seed)
    print(f"Camera motion: {camera_motion} (variant: {args.camera_variant})")
    animate_camera_motion(camera, camera.location, look_at, num_frames,
                         motion_type=camera_motion, seed=render_seed, camera_variant=args.camera_variant)

    # Animate objects
    print("Animating objects...")
    # PKL pose data is always in Y-up coordinates
    # We reset FBX objects to identity, so they can receive Y-up transforms like normal
    animate_objects(objects_data, object_pose_params, num_frames, y_up=args.y_up)

    # Render with human
    print("Rendering with human...")
    if args.use_characters:
        # Show all character meshes
        for mesh in character_meshes:
            mesh.hide_render = False
    else:
        human_obj.hide_render = False
    output_with = os.path.join(output_dir, f"{output_name}_with_human")
    frames_dir_with = render_video(output_with, num_frames, scene)

    # NEW: Render background with original animations but no human
    # This lets us compute true scene change (what changed underneath the human)
    print("Rendering background with original animations (no human)...")
    if args.use_characters:
        # Hide all character meshes
        for mesh in character_meshes:
            mesh.hide_render = True
    else:
        human_obj.hide_render = True

    output_background = os.path.join(output_dir, f"{output_name}_background_original")
    frames_dir_background = render_video(output_background, num_frames, scene)

    # Show human again for subsequent renders
    if args.use_characters:
        for mesh in character_meshes:
            mesh.hide_render = False
    else:
        human_obj.hide_render = False

    # Render without human (with physics applied - objects fall/move)
    print("Rendering without human (with physics)...")
    if args.use_characters:
        # Hide all character meshes
        for mesh in character_meshes:
            mesh.hide_render = True
    else:
        human_obj.hide_render = True

    if args.enable_physics:
        print("  Enabling physics simulation (with manual configuration)...")

        # Load physics configuration for this sequence
        physics_config = load_physics_config(sequence_name)

        if physics_config:
            static_objects = set(physics_config.get('static_objects', []))
            physics_objects = set(physics_config.get('physics_objects', []))
            print(f"  Loaded manual physics config:")
            print(f"    Static objects: {len(static_objects)} - {sorted(static_objects)}")
            print(f"    Physics objects: {len(physics_objects)} - {sorted(physics_objects)}")
        else:
            print(f"  No manual config found - applying physics to ALL objects")
            static_objects = set()
            physics_objects = set(objects_data.keys())

        # Ensure rigid body world exists
        if not scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        # Enable physics for ground
        ground.hide_render = False
        if not ground.rigid_body:
            bpy.context.view_layer.objects.active = ground
            bpy.ops.rigidbody.object_add()
        ground.rigid_body.type = 'PASSIVE'
        ground.rigid_body.collision_shape = 'MESH'
        ground.rigid_body.friction = 1.0
        ground.rigid_body.restitution = 0.1

        # Enable physics for walls
        for wall in walls:
            if not wall.rigid_body:
                bpy.context.view_layer.objects.active = wall
                bpy.ops.rigidbody.object_add()
            wall.rigid_body.type = 'PASSIVE'
            wall.rigid_body.collision_shape = 'MESH'
            wall.rigid_body.friction = 1.0

        # Process each object based on manual configuration
        physics_count = 0
        static_count = 0

        for obj_name, obj_info in objects_data.items():
            obj = obj_info['object']

            # Determine if this object should have physics
            should_have_physics = obj_name in physics_objects if physics_config else True

            # Set to initial position from frame 1
            if obj_name in object_pose_params:
                initial_pose = object_pose_params[obj_name][0]
                w, x, y, z = float(initial_pose[0]), float(initial_pose[1]), float(initial_pose[2]), float(initial_pose[3])
                tx, ty, tz = float(initial_pose[4]), float(initial_pose[5]), float(initial_pose[6])

                # Transform position from Y-up to Z-up (Blender)
                if args.y_up:
                    tx, ty, tz = tx, tz, ty

                obj.location = (tx, ty, tz)

                # Transform quaternion rotation
                if args.y_up:
                    q_bl = quat_yup_to_blender((w, x, y, z))
                else:
                    q_bl = mathutils.Quaternion((w, x, y, z))

                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = q_bl

            if should_have_physics:
                # Enable ACTIVE physics (object can move)
                # Clear all animation keyframes (physics will control motion)
                if obj.animation_data:
                    obj.animation_data_clear()

                # Enable rigid body physics
                bpy.context.view_layer.objects.active = obj
                if not obj.rigid_body:
                    bpy.ops.rigidbody.object_add()

                obj.rigid_body.type = 'ACTIVE'
                obj.rigid_body.collision_shape = 'CONVEX_HULL'
                obj.rigid_body.mass = 2.0
                obj.rigid_body.friction = 1.0
                obj.rigid_body.restitution = 0.1
                obj.rigid_body.linear_damping = 0.04
                obj.rigid_body.angular_damping = 0.1
                obj.rigid_body.use_deactivation = True

                physics_count += 1
            else:
                # Keep object STATIC (no physics, stays in initial position)
                # Clear animation and freeze at frame 1
                if obj.animation_data:
                    obj.animation_data_clear()

                # Enable as PASSIVE rigid body for collision detection
                bpy.context.view_layer.objects.active = obj
                if not obj.rigid_body:
                    bpy.ops.rigidbody.object_add()

                obj.rigid_body.type = 'PASSIVE'
                obj.rigid_body.collision_shape = 'MESH'
                obj.rigid_body.friction = 1.0

                static_count += 1

        # Set up rigid body world settings
        scene.rigidbody_world.point_cache.frame_start = 1
        scene.rigidbody_world.point_cache.frame_end = num_frames

        # Bake physics simulation
        print(f"  Baking physics: {physics_count} active objects, {static_count} static objects...")
        bpy.ops.ptcache.bake_all(bake=True)
    else:
        # Freeze objects at their initial position (frame 1) - no human means no interaction
        for obj_name, obj_info in objects_data.items():
            obj = obj_info['object']

            # Clear all animation keyframes
            if obj.animation_data:
                obj.animation_data_clear()

            # Set to initial position from frame 1
            if obj_name in object_pose_params:
                initial_pose = object_pose_params[obj_name][0]  # First frame
                w, x, y, z = float(initial_pose[0]), float(initial_pose[1]), float(initial_pose[2]), float(initial_pose[3])
                tx, ty, tz = float(initial_pose[4]), float(initial_pose[5]), float(initial_pose[6])

                # Transform position from Y-up to Z-up (Blender)
                if args.y_up:
                    tx, ty, tz = tx, tz, ty

                obj.location = (tx, ty, tz)

                # Transform quaternion rotation
                if args.y_up:
                    q_bl = quat_yup_to_blender((w, x, y, z))
                else:
                    q_bl = mathutils.Quaternion((w, x, y, z))

                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = q_bl

    output_without = os.path.join(output_dir, f"{output_name}_without_human")
    frames_dir_without = render_video(output_without, num_frames, scene)

    # --- Rendering human mask (OCCLUSION-AWARE) ---
    print("Rendering occlusion-aware human mask...")

    # KEEP EVERYTHING VISIBLE - objects will occlude the human naturally
    if args.use_characters:
        for mesh in character_meshes:
            mesh.hide_render = False
    else:
        human_obj.hide_render = False
    for obj in objects_data.values():
        obj['object'].hide_render = False  # KEEP VISIBLE for occlusion
    ground.hide_render = False  # KEEP VISIBLE
    for w in walls:
        w.hide_render = False  # KEEP VISIBLE

    # Save originals and apply white emission material to human
    original_materials_dict = {}
    if args.use_characters:
        # Save original materials for all character meshes
        for mesh in character_meshes:
            original_materials_dict[mesh.name] = list(mesh.data.materials)
    else:
        original_materials_dict[human_obj.name] = list(human_obj.data.materials)

    # Save original materials for objects, ground, walls
    objects_original_materials = {}
    for obj_name, obj_data in objects_data.items():
        objects_original_materials[obj_name] = list(obj_data['object'].data.materials)
    ground_original_materials = list(ground.data.materials)
    walls_original_materials = [list(w.data.materials) for w in walls]

    orig_world_nodes = None
    if scene.world and scene.world.use_nodes:
        orig_world_nodes = scene.world.node_tree.copy()

    # Force human/character to pure white emission
    mask_mat = bpy.data.materials.new(name="TempMask_Material")
    mask_mat.use_nodes = True
    nodes = mask_mat.node_tree.nodes
    links = mask_mat.node_tree.links
    nodes.clear()
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1, 1, 1, 1)   # white
    emission.inputs['Strength'].default_value = 10.0        # bright
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output.inputs['Surface'])

    if args.use_characters:
        for mesh in character_meshes:
            mesh.data.materials.clear()
            mesh.data.materials.append(mask_mat)
    else:
        human_obj.data.materials.clear()
        human_obj.data.materials.append(mask_mat)

    # Make ALL objects, ground, and walls BLACK so they occlude the white human
    black_mat = bpy.data.materials.new(name="TempBlack_Material")
    black_mat.use_nodes = True
    nodes_black = black_mat.node_tree.nodes
    links_black = black_mat.node_tree.links
    nodes_black.clear()
    emission_black = nodes_black.new('ShaderNodeEmission')
    emission_black.inputs['Color'].default_value = (0, 0, 0, 1)   # black
    emission_black.inputs['Strength'].default_value = 10.0
    output_black = nodes_black.new('ShaderNodeOutputMaterial')
    links_black.new(emission_black.outputs['Emission'], output_black.inputs['Surface'])

    # Apply black material to all objects
    for obj_name, obj_data in objects_data.items():
        obj_data['object'].data.materials.clear()
        obj_data['object'].data.materials.append(black_mat)

    # Apply black material to ground
    ground.data.materials.clear()
    ground.data.materials.append(black_mat)

    # Apply black material to walls
    for w in walls:
        w.data.materials.clear()
        w.data.materials.append(black_mat)

    # Black world background (no transparency, no alpha)
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = 'RGB'
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0, 0, 0, 1)
        bg.inputs['Strength'].default_value = 1.0

    # Tone mapping: avoid Filmic lifting blacks
    orig_view_transform = scene.view_settings.view_transform
    orig_exposure = scene.view_settings.exposure
    orig_gamma = scene.view_settings.gamma
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    # Turn off TAA/motion blur to kill mask fringes
    orig_taa = scene.eevee.use_taa_reprojection if hasattr(scene.eevee, 'use_taa_reprojection') else False
    orig_motion_blur = scene.render.use_motion_blur
    if hasattr(scene.eevee, 'use_taa_reprojection'):
        scene.eevee.use_taa_reprojection = False
    scene.render.use_motion_blur = False

    # Render mask frames
    output_human_mask = os.path.join(output_dir, f"{output_name}_human_only")
    frames_dir_human_mask = os.path.join(output_dir, f"{output_name}_human_mask_frames")
    os.makedirs(frames_dir_human_mask, exist_ok=True)
    for frame in range(1, num_frames + 1):
        scene.frame_set(frame)
        scene.render.filepath = os.path.join(frames_dir_human_mask, f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)

    # Restore scene materials
    if args.use_characters:
        for mesh in character_meshes:
            mesh.data.materials.clear()
            for mat in original_materials_dict[mesh.name]:
                mesh.data.materials.append(mat)
    else:
        human_obj.data.materials.clear()
        for mat in original_materials_dict[human_obj.name]:
            human_obj.data.materials.append(mat)

    # Restore object materials
    for obj_name, obj_data in objects_data.items():
        obj_data['object'].data.materials.clear()
        for mat in objects_original_materials[obj_name]:
            obj_data['object'].data.materials.append(mat)

    # Restore ground materials
    ground.data.materials.clear()
    for mat in ground_original_materials:
        ground.data.materials.append(mat)

    # Restore wall materials
    for i, w in enumerate(walls):
        w.data.materials.clear()
        for mat in walls_original_materials[i]:
            w.data.materials.append(mat)

    # Cleanup temporary materials
    bpy.data.materials.remove(mask_mat)
    bpy.data.materials.remove(black_mat)

    # Restore world nodes if we had them
    if orig_world_nodes is not None:
        scene.world.node_tree.nodes.clear()
        scene.world.node_tree.links.clear()
        for node in orig_world_nodes.nodes:
            new_node = scene.world.node_tree.nodes.new(node.bl_idname)
            for attr in dir(node):
                if not attr.startswith('_') and attr not in ['location', 'width', 'height']:
                    try:
                        setattr(new_node, attr, getattr(node, attr))
                    except:
                        pass

    scene.view_settings.view_transform = orig_view_transform
    scene.view_settings.exposure = orig_exposure
    scene.view_settings.gamma = orig_gamma
    if hasattr(scene.eevee, 'use_taa_reprojection'):
        scene.eevee.use_taa_reprojection = orig_taa
    scene.render.use_motion_blur = orig_motion_blur
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    for obj in objects_data.values():
        obj['object'].hide_render = False
    ground.hide_render = False
    for w in walls:
        w.hide_render = False

    # Generate quad-mask from the four renders
    print("Generating quad-mask...")
    frames_dir_mask = os.path.join(output_dir, f"{output_name}_mask_frames")
    create_trimask_from_frames(frames_dir_with, frames_dir_background, frames_dir_without, frames_dir_human_mask,
                               frames_dir_mask, threshold=8, human_dilate_px=2, change_erode_px=0)

    # Create output subdirectory for this scenario
    scenario_output_dir = os.path.join(output_dir, output_name)
    os.makedirs(scenario_output_dir, exist_ok=True)

    # Convert to videos with new naming scheme
    print("Converting RGB videos...")
    success_with = frames_to_video(frames_dir_with,
                                   os.path.join(scenario_output_dir, "rgb_full.mp4"),
                                   fps=args.fps)
    print(f"  rgb_full.mp4: {'SUCCESS' if success_with else 'FAILED'}")

    success_without = frames_to_video(frames_dir_without,
                                     os.path.join(scenario_output_dir, "rgb_removed.mp4"),
                                     fps=args.fps)
    print(f"  rgb_removed.mp4: {'SUCCESS' if success_without else 'FAILED'}")

    # Convert 3 mask types to videos
    print("Converting mask videos...")
    frames_dir_human_mask_binary = frames_dir_mask + "_human"
    frames_dir_change_mask_binary = frames_dir_mask + "_change"

    print(f"  Looking for human mask frames in: {frames_dir_human_mask_binary}")
    print(f"  Directory exists: {os.path.exists(frames_dir_human_mask_binary)}")
    if os.path.exists(frames_dir_human_mask_binary):
        num_frames = len([f for f in os.listdir(frames_dir_human_mask_binary) if f.endswith('.png')])
        print(f"  Found {num_frames} frames")

    success_human_mask = frames_to_video(frames_dir_human_mask_binary,
                                         os.path.join(scenario_output_dir, "mask_human.mp4"),
                                         fps=args.fps,
                                         is_mask=True)
    print(f"  mask_human.mp4: {'SUCCESS' if success_human_mask else 'FAILED'}")

    print(f"  Looking for change mask frames in: {frames_dir_change_mask_binary}")
    print(f"  Directory exists: {os.path.exists(frames_dir_change_mask_binary)}")
    if os.path.exists(frames_dir_change_mask_binary):
        num_frames = len([f for f in os.listdir(frames_dir_change_mask_binary) if f.endswith('.png')])
        print(f"  Found {num_frames} frames")

    success_change_mask = frames_to_video(frames_dir_change_mask_binary,
                                          os.path.join(scenario_output_dir, "mask_change.mp4"),
                                          fps=args.fps,
                                          is_mask=True)
    print(f"  mask_change.mp4: {'SUCCESS' if success_change_mask else 'FAILED'}")

    success_quad_mask = frames_to_video(frames_dir_mask,
                                       os.path.join(scenario_output_dir, "mask.mp4"),
                                       fps=args.fps,
                                       is_mask=True)
    print(f"  mask.mp4: {'SUCCESS' if success_quad_mask else 'FAILED'}")

    # Cleanup frame directories
    import shutil
    shutil.rmtree(frames_dir_with, ignore_errors=True)
    shutil.rmtree(frames_dir_background, ignore_errors=True)
    shutil.rmtree(frames_dir_without, ignore_errors=True)
    shutil.rmtree(frames_dir_human_mask, ignore_errors=True)
    shutil.rmtree(frames_dir_mask, ignore_errors=True)
    shutil.rmtree(frames_dir_human_mask_binary, ignore_errors=True)
    shutil.rmtree(frames_dir_change_mask_binary, ignore_errors=True)

    if success_with and success_without and success_human_mask and success_change_mask and success_quad_mask:
        print(f"Successfully created video set for {output_name}")
        print(f"  RGB videos: rgb_full.mp4, rgb_removed.mp4")
        print(f"  Mask videos: mask_human.mp4, mask_change.mp4, mask.mp4")
        print(f"  Output directory: {scenario_output_dir}")
        return True
    else:
        print(f"Failed to create videos for {output_name}")
        return False


def main():
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process sequences
    successful = 0
    failed = 0

    for seq_name in args.sequences:
        seq_path = os.path.join(args.dataset_dir, seq_name)
        if not os.path.exists(seq_path):
            print(f"Warning: Sequence {seq_name} not found at {seq_path}")
            failed += 1
            continue

        if process_sequence(seq_path, args.output_dir, args):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()