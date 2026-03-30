#!/usr/bin/env python3
# ENHANCED VERSION: Variable object count (2-5), larger size variation, multi-object removal
# - 2-3 objects: Remove 1 object
# - 4-5 objects: Remove 2 objects
# - Much larger size variation (3x to 20x)

import os, shutil, tempfile, logging, glob, subprocess, math, gc
from pathlib import Path
import numpy as np

import imageio.v2 as iio
from PIL import Image
from types import SimpleNamespace as NS

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import pybullet as pb

# Configuration - HUGE SIZE VARIATION
SCALE_RANGE = (5.0, 50.0)  # From small (3x) to MASSIVE (20x)!
DEFAULT_SPP = 32
DBG = True

# List of HDRIs to avoid (too dark)
DARK_BACKGROUNDS = [
    'abandoned_hall_01', 'adams_place_bridge', 'autoshop_01',
    'bathroom', 'carpentry_shop_02', 'castle_corridor', 'castle_hall',
    'coffee_shop_1k', 'conference_room', 'courtyard_night', 'dining_room',
    'empty_workshop', 'entrance_hall', 'fireplace', 'garage',
    'hall_of_mammals', 'hotel_room', 'industrial_pipe_and_valve_01',
    'industrial_sunset_puresky', 'lebombo', 'machine_shop_01',
    'museum_of_ethnography', 'narrow_moonlit_road', 'night_bridge',
    'old_depot', 'old_hall', 'old_room', 'peppermint_powerplant',
    'phone_shop_01', 'photo_studio_01', 'piazza_san_marco',
    'preller_drive', 'reading_room', 'red_wall', 'royal_esplanade',
    'secluded_beach', 'shudu_lake', 'small_empty_house', 'solitude_interior',
    'st_fagans_interior', 'studio_small_03', 'syferfontein_0d',
    'the_sky_is_on_fire', 'thatch_chapel', 'theater_01', 'theater_02',
    'tiergarten', 'urban_alley_01', 'urban_courtyard', 'urban_street_04',
    'venice_sunset', 'veranda', 'vintage_measuring_lab', 'warehouse',
    'workshop', 'zen_garden'
]

def cleanup_scene():
    """Clean up Blender and PyBullet to prevent memory leaks"""
    try:
        pb.disconnect()
    except:
        pass
    gc.collect()
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    except:
        pass

def apply_camera_motion(camera, motion_type, cam_distance, cam_angle, cam_height,
                        look_at_point, frame_start, frame_end, rng, motion_params=None):
    """Apply camera motion animation based on motion type.

    Args:
        camera: Kubric camera object
        motion_type: One of ['static', 'zoom_in', 'zoom_out', 'pan_left', 'pan_right']
        cam_distance: Initial camera distance from center
        cam_angle: Initial camera angle (radians)
        cam_height: Camera height
        look_at_point: Point camera looks at (tuple)
        frame_start: First frame
        frame_end: Last frame
        rng: Random number generator
        motion_params: Pre-computed motion parameters (for reproducing exact motion)

    Returns:
        Dictionary of motion parameters used (for reusing in other passes)
    """
    if motion_type == 'static':
        # No animation, already set
        return {}

    # Calculate initial position
    pos_x_start = cam_distance * np.cos(cam_angle)
    pos_y_start = -cam_distance * np.sin(cam_angle)

    # Use pre-computed parameters or generate new ones
    if motion_params is None:
        motion_params = {}
        if motion_type == 'zoom_in':
            motion_params['zoom_factor'] = rng.uniform(0.6, 0.8)
        elif motion_type == 'zoom_out':
            motion_params['zoom_factor'] = rng.uniform(1.3, 1.5)
        elif motion_type in ['pan_left', 'pan_right']:
            motion_params['angle_change'] = rng.uniform(20, 40) * np.pi / 180

    if motion_type == 'zoom_in':
        # Move camera closer over time (reduce distance by 20-40%)
        cam_distance_end = cam_distance * motion_params['zoom_factor']
        pos_x_end = cam_distance_end * np.cos(cam_angle)
        pos_y_end = -cam_distance_end * np.sin(cam_angle)

    elif motion_type == 'zoom_out':
        # Move camera farther over time (increase distance by 30-50%)
        cam_distance_end = cam_distance * motion_params['zoom_factor']
        pos_x_end = cam_distance_end * np.cos(cam_angle)
        pos_y_end = -cam_distance_end * np.sin(cam_angle)

    elif motion_type == 'pan_left':
        # Rotate camera angle to the left (increase angle by 20-40 degrees)
        cam_angle_end = cam_angle + motion_params['angle_change']
        pos_x_end = cam_distance * np.cos(cam_angle_end)
        pos_y_end = -cam_distance * np.sin(cam_angle_end)

    elif motion_type == 'pan_right':
        # Rotate camera angle to the right (decrease angle by 20-40 degrees)
        cam_angle_end = cam_angle - motion_params['angle_change']
        pos_x_end = cam_distance * np.cos(cam_angle_end)
        pos_y_end = -cam_distance * np.sin(cam_angle_end)

    # Set keyframes for smooth animation
    camera.position = (pos_x_start, pos_y_start, cam_height)
    camera.look_at(look_at_point)
    camera.keyframe_insert("position", frame_start)
    camera.keyframe_insert("quaternion", frame_start)

    camera.position = (pos_x_end, pos_y_end, cam_height)
    camera.look_at(look_at_point)
    camera.keyframe_insert("position", frame_end)
    camera.keyframe_insert("quaternion", frame_end)

    return motion_params

def ffmpeg_exe():
    import shutil as _sh
    ff = _sh.which("ffmpeg")
    if ff: return ff
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def encode_sequence(pattern_glob: str, out_mp4: Path, fps: int, lossless: bool = False):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe()

    if lossless:
        # Use H.264 lossless mode (qp=0) for lossless encoding (preserves exact pixel values)
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-framerate", str(fps),
               "-pattern_type", "glob", "-i", pattern_glob,
               "-c:v", "libx264", "-qp", "0",
               "-pix_fmt", "gray",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               str(out_mp4)]
    else:
        # Standard lossy encoding for RGB videos
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
               "-framerate", str(fps),
               "-pattern_type", "glob", "-i", pattern_glob,
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               str(out_mp4)]
    subprocess.run(cmd, check=True)

def generate_one(rng, out_dir: Path,
                 kubasic_source, gso_source, hdri_source,
                 objects_split="train", backgrounds_split="train",
                 frame_start=1, frame_end=40, frame_rate=12,
                 resolution=256, step_rate=240, spp=DEFAULT_SPP):

    print(f"\n[DEBUG] ===== GENERATING SCENE: {out_dir.name} =====")

    # RANDOMIZE NUMBER OF OBJECTS (2-5)
    num_objects = rng.choice([3, 3, 4, 4, 5, 6, 6, 7, 7])  # Weighted toward 3-4 objects
    num_to_remove = 2 if num_objects >= 4 else 1

    print(f"[DEBUG] Scene will have {num_objects} objects, removing {num_to_remove}")

    tmp_scratch = tempfile.mkdtemp(prefix="kb_scratch_")
    FLAGS_like = NS(
        frame_start=frame_start,
        frame_end=frame_end,
        frame_rate=frame_rate,
        step_rate=step_rate,
        resolution=resolution,
        job_dir=str(out_dir.parent),
        scratch_dir=tmp_scratch,
        seed=12,
        logging_level="WARNING",
    )

    tmpA = tmpB = tmpC = tmpMask = None
    try:
        cleanup_scene()

        # ========== PASS A: All objects (full physics, all visible) ==========
        print(f"[DEBUG] Creating Pass A with {num_objects} objects...")

        scene, _, output_dir, scratch_dir = kb.setup(FLAGS_like)
        renderer = Blender(scene, scratch_dir, samples_per_pixel=int(spp))
        simulator = PyBullet(scene, scratch_dir)

        # Background (bright only)
        train_bg, test_bg = hdri_source.get_test_split(fraction=0.1)
        bg_candidates = train_bg if backgrounds_split=="train" else test_bg
        bright_backgrounds = [bg for bg in bg_candidates if not any(dark in bg for dark in DARK_BACKGROUNDS)]
        if not bright_backgrounds:
            bright_backgrounds = bg_candidates

        hdri_id = rng.choice(bright_backgrounds)
        bg_tex = hdri_source.create(asset_id=hdri_id)
        scene.metadata["background"] = hdri_id
        renderer._set_ambient_light_hdri(bg_tex.filename)
        print(f"[DEBUG] Background: {hdri_id}")

        # Dome (floor)
        dome = kubasic_source.create(asset_id="dome", name="dome",
                                     friction=0.5, restitution=0.2,
                                     static=True, background=True)
        scene += dome
        dome_b = dome.linked_objects[renderer]
        dome_b.data.materials[0].node_tree.nodes["Image Texture"].image = bpy.data.images.load(bg_tex.filename)

        # Camera - adaptive distance based on max object scale
        max_expected_scale = 15.0  # Rough average of our range
        cam_distance = rng.uniform(12.0, 20.0)  # Further back for huge objects
        cam_angle = rng.uniform(-30, 30) * np.pi / 180
        cam_height = rng.uniform(5.0, 10.0)
        look_at_point = (0, 0, 2.0)

        # Randomly choose camera motion type
        camera_motion_type = rng.choice(['static', 'zoom_in', 'zoom_out', 'pan_left', 'pan_right'])
        print(f"[DEBUG] Camera motion: {camera_motion_type}")

        scene.camera = kb.PerspectiveCamera(focal_length=35.0, sensor_width=32)
        scene.camera.position = (cam_distance * np.cos(cam_angle),
                                 -cam_distance * np.sin(cam_angle),
                                 cam_height)
        scene.camera.look_at(look_at_point)

        # Apply camera motion animation and capture parameters for Pass B
        camera_motion_params = apply_camera_motion(scene.camera, camera_motion_type, cam_distance, cam_angle,
                           cam_height, look_at_point, scene.frame_start, scene.frame_end, rng)

        # Get object IDs
        train_split, test_split = gso_source.get_test_split(fraction=0.1)
        gso_source.asset_ids = train_split if objects_split=="train" else test_split

        # ===== CREATE OBJECTS =====
        objects = []
        initial_states = []

        # First object: TARGET (at rest, will be hit)
        aid = rng.choice(gso_source.asset_ids)
        target = gso_source.create(asset_id=aid, name="object_0_target")
        target.static = False
        target.mass = rng.uniform(0.4, 1.0)
        target.friction = 0.4
        target.restitution = rng.uniform(0.2, 0.4)

        # SIZE VARIATION: Some objects small, some HUGE
        if rng.random() < 0.3:  # 30% chance of extra large
            target.scale = rng.uniform(12.0, 20.0)
        else:
            target.scale = rng.uniform(3.0, 12.0)
        print(f"[DEBUG] Target (obj 0) scale: {target.scale}x")

        scene += target
        target.position = (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), 1.5)
        target.quaternion = (1, 0, 0, 0)
        objects.append(target)

        # Objects to remove (indices)
        objects_to_remove = []

        # Create remaining objects
        for i in range(1, num_objects):
            oid = rng.choice(gso_source.asset_ids)
            obj = gso_source.create(asset_id=oid, name=f"object_{i}")
            obj.static = False
            obj.mass = rng.uniform(0.6, 2.0)
            obj.friction = rng.uniform(0.3, 0.5)
            obj.restitution = rng.uniform(0.2, 0.5)

            # SIZE VARIATION
            if rng.random() < 0.3:  # 30% chance of extra large
                obj.scale = rng.uniform(12.0, 20.0)
            else:
                obj.scale = rng.uniform(3.0, 12.0)
            print(f"[DEBUG] Object {i} scale: {obj.scale}x")

            scene += obj

            # Position objects in a circle around target
            angle = (i - 1) * 2 * np.pi / (num_objects - 1)
            distance = rng.uniform(4.0, 7.0)
            height = rng.uniform(1.0, 3.0)

            obj.position = (target.position[0] + distance * np.cos(angle),
                           target.position[1] + distance * np.sin(angle),
                           height)

            objects.append(obj)

            # Mark some objects for removal
            if i <= num_to_remove:
                objects_to_remove.append(i)
                print(f"[DEBUG] Object {i} will be REMOVED in Pass B")

        # Initialize physics
        _ = simulator.run(frame_start=scene.frame_start, frame_end=scene.frame_start)

        # Set velocities based on object roles
        # Target stays at rest
        objects[0].velocity = (0.0, 0.0, 0.0)
        objects[0].angular_velocity = (0.0, 0.0, 0.0)

        # Others move with varied strategies
        for i in range(1, num_objects):
            obj = objects[i]

            if i in objects_to_remove:
                # Objects to be removed should hit the target
                speed = rng.uniform(3.0, 9.0)
                to_target = np.array(target.position) - np.array(obj.position)
                to_target[2] = 0  # Mostly horizontal
                to_target = to_target / (np.linalg.norm(to_target) + 1e-6)
                obj.velocity = tuple(to_target * speed)
            else:
                # Objects that remain should miss or have different motion
                speed = rng.uniform(3.0, 7.0)
                # Random direction, possibly missing
                miss_offset = rng.uniform(-3, 3, size=2)
                miss_target = np.array([target.position[0] + miss_offset[0],
                                        target.position[1] + miss_offset[1],
                                        1.5])
                to_miss = miss_target - np.array(obj.position)
                to_miss[2] = 0
                to_miss = to_miss / (np.linalg.norm(to_miss) + 1e-6)
                obj.velocity = tuple(to_miss * speed)

            obj.angular_velocity = (0.0, 0.0, 0.0)
            print(f"[DEBUG] Object {i} velocity: {[f'{v:.2f}' for v in obj.velocity]}")

        # Save initial states
        for obj in objects:
            initial_states.append({
                'position': tuple(obj.position),
                'quaternion': tuple(obj.quaternion),
                'velocity': tuple(obj.velocity) if hasattr(obj, 'velocity') else (0,0,0),
                'angular_velocity': tuple(obj.angular_velocity) if hasattr(obj, 'angular_velocity') else (0,0,0),
                'asset_id': obj.asset_id,
                'scale': obj.scale,
                'mass': obj.mass,
                'friction': obj.friction,
                'restitution': obj.restitution
            })

        # Run simulation
        print("[DEBUG] Running Pass A simulation...")
        collisions_data, collision_list = simulator.run(
            frame_start=scene.frame_start,
            frame_end=scene.frame_end
        )

        # Check collisions with target
        processed_colls = kb.process_collisions(collision_list, scene, assets_subset=objects)
        target_hit = False
        if processed_colls and 'collisions' in processed_colls:
            for coll in processed_colls['collisions']:
                if 'instances' in coll:
                    names = [str(x.name if hasattr(x, 'name') else x) for x in coll['instances']]
                    if objects[0].name in names:
                        for removed_idx in objects_to_remove:
                            if objects[removed_idx].name in names:
                                target_hit = True
                                print(f"[DEBUG] Object {removed_idx} hit the target!")

        # Render Pass A
        print("[DEBUG] Rendering Pass A...")
        data_A = renderer.render()
        tmpA = Path(tempfile.mkdtemp(prefix="passA_"))

        kb.compute_visibility(data_A["segmentation"], scene.assets)
        vis_assets = [a for a in scene.foreground_assets if np.max(a.metadata.get("visibility", [0])) > 0]

        for obj in objects:
            if obj not in vis_assets:
                vis_assets.append(obj)
                obj.metadata["visibility"] = [0.01]

        vis_assets = sorted(vis_assets, key=lambda a: np.sum(a.metadata.get("visibility", [0])), reverse=True)
        data_A["segmentation"] = kb.adjust_segmentation_idxs(data_A["segmentation"], scene.assets, vis_assets)
        kb.write_image_dict(data_A, tmpA)

        # Get IDs of objects to remove for mask
        inst_ids = {asset:(i+1) for i, asset in enumerate(vis_assets)}
        removed_ids = [inst_ids.get(objects[idx], None) for idx in objects_to_remove]
        removed_ids = [rid for rid in removed_ids if rid is not None]

        # ========== PASS C: Same physics as A, but removed objects invisible ==========
        print(f"[DEBUG] Creating Pass C (removed objects invisible but physically present)...")

        tmpC = Path(tempfile.mkdtemp(prefix="passC_"))

        # Make removed objects invisible in Blender (but they're still in physics)
        for idx in objects_to_remove:
            obj = objects[idx]
            blender_obj = obj.linked_objects[renderer]
            blender_obj.hide_render = True  # Invisible in render, but physics already baked

        # Re-render the same scene with removed objects hidden
        print("[DEBUG] Rendering Pass C...")
        data_C = renderer.render()

        # Use same visibility assets but filter out removed objects
        vis_assets_C = [a for a in vis_assets if a not in [objects[idx] for idx in objects_to_remove]]
        data_C["segmentation"] = kb.adjust_segmentation_idxs(data_C["segmentation"], scene.assets, vis_assets_C)
        kb.write_image_dict(data_C, tmpC)

        # Make objects visible again (for cleanup)
        for idx in objects_to_remove:
            obj = objects[idx]
            blender_obj = obj.linked_objects[renderer]
            blender_obj.hide_render = False

        # Clean up Pass A/C (shared scene)
        del renderer
        del simulator
        del scene
        cleanup_scene()

        # ========== PASS B: Without removed objects ==========
        print(f"[DEBUG] Creating Pass B without objects {objects_to_remove}...")

        tmpB = Path(tempfile.mkdtemp(prefix="passB_"))
        sceneB, _, _, scratch_dirB = kb.setup(FLAGS_like)
        rendererB = Blender(sceneB, scratch_dirB, samples_per_pixel=int(spp))
        simulatorB = PyBullet(sceneB, scratch_dirB)

        # Same background
        bg_texB = hdri_source.create(asset_id=hdri_id)
        sceneB.metadata["background"] = hdri_id
        rendererB._set_ambient_light_hdri(bg_texB.filename)

        # Same dome
        domeB = kubasic_source.create(asset_id="dome", name="dome",
                                      friction=0.5, restitution=0.2,
                                      static=True, background=True)
        sceneB += domeB
        domeB_b = domeB.linked_objects[rendererB]
        domeB_b.data.materials[0].node_tree.nodes["Image Texture"].image = bpy.data.images.load(bg_texB.filename)

        # Same camera with same motion
        sceneB.camera = kb.PerspectiveCamera(focal_length=35.0, sensor_width=32)
        sceneB.camera.position = (cam_distance * np.cos(cam_angle),
                                  -cam_distance * np.sin(cam_angle),
                                  cam_height)
        sceneB.camera.look_at(look_at_point)

        # Apply same camera motion animation as Pass A (reuse exact parameters)
        apply_camera_motion(sceneB.camera, camera_motion_type, cam_distance, cam_angle,
                           cam_height, look_at_point, sceneB.frame_start, sceneB.frame_end, rng, camera_motion_params)

        # Recreate objects except the removed ones
        objectsB = []
        for i in range(num_objects):
            if i not in objects_to_remove:
                state = initial_states[i]
                objB = gso_source.create(asset_id=state['asset_id'], name=f"object_{i}_B")
                objB.static = False if i > 0 else False  # Target can still be dynamic
                objB.mass = state['mass']
                objB.friction = state['friction']
                objB.restitution = state['restitution']
                objB.scale = state['scale']
                objB.position = state['position']
                objB.quaternion = state['quaternion']
                sceneB += objB
                objectsB.append((i, objB))

        # Initialize physics
        _ = simulatorB.run(frame_start=sceneB.frame_start, frame_end=sceneB.frame_start)

        # Set velocities for remaining objects
        for orig_idx, objB in objectsB:
            if orig_idx == 0:
                # Target stays at rest (no objects to hit it)
                objB.velocity = (0.0, 0.0, 0.0)
                objB.angular_velocity = (0.0, 0.0, 0.0)
            else:
                # Other objects keep their original velocities
                objB.velocity = initial_states[orig_idx]['velocity']
                objB.angular_velocity = initial_states[orig_idx]['angular_velocity']

        # Run simulation
        print("[DEBUG] Running Pass B simulation...")
        simulatorB.run(frame_start=sceneB.frame_start, frame_end=sceneB.frame_end)

        # Render Pass B
        print("[DEBUG] Rendering Pass B...")
        data_B = rendererB.render()
        kb.compute_visibility(data_B["segmentation"], sceneB.assets)

        # Save visible assets for Pass B mask computation
        vis_assetsB = [a for a in sceneB.foreground_assets if np.max(a.metadata.get("visibility", [0])) > 0]
        for obj_idx, objB in objectsB:
            if objB not in vis_assetsB:
                vis_assetsB.append(objB)
                objB.metadata["visibility"] = [0.01]
        vis_assetsB = sorted(vis_assetsB, key=lambda a: np.sum(a.metadata.get("visibility", [0])), reverse=True)
        data_B["segmentation"] = kb.adjust_segmentation_idxs(data_B["segmentation"], sceneB.assets, vis_assetsB)

        kb.write_image_dict(data_B, tmpB)

        # Clean up Pass B
        del rendererB
        del simulatorB
        del sceneB
        cleanup_scene()

        # ========== CREATE MASKS ==========
        print("[DEBUG] Creating masks with overlap detection using 3 passes...")

        rgbaA_paths = sorted(glob.glob(str(tmpA / "rgba_*.png")))
        rgbaC_paths = sorted(glob.glob(str(tmpC / "rgba_*.png")))
        rgbaB_paths = sorted(glob.glob(str(tmpB / "rgba_*.png")))
        segA_paths = sorted(glob.glob(str(tmpA / "segmentation_*.png")))

        tmpMask = Path(tempfile.mkdtemp(prefix="mask_"))
        tmpMask.mkdir(exist_ok=True, parents=True)

        for pA, pC, pB, pSA in zip(rgbaA_paths, rgbaC_paths, rgbaB_paths, segA_paths):
            A = iio.imread(pA)[...,:3]
            C = iio.imread(pC)[...,:3]
            B = iio.imread(pB)[...,:3]
            with Image.open(pSA) as im:
                SA = np.array(im)
            if SA.ndim == 3: SA = SA[...,0]

            H, W = A.shape[:2]
            m = np.full((H, W), 255, np.uint8)

            # Step 1: Black mask - from Pass A segmentation (solid object pixels only, no shadows)
            black_mask = np.zeros((H, W), dtype=bool)
            for rid in removed_ids:
                black_mask |= (SA == rid)

            # Step 2: Grey mask - C vs B RGB difference (physics changes including shadows)
            # Where remaining objects moved differently (C has same physics as A, B has altered physics)
            diff_CB = np.abs(C.astype(np.int16) - B.astype(np.int16)).max(axis=-1)
            grey_mask = (diff_CB >= 15)  # Threshold for visible difference

            # Step 3: Overlap - where black and grey intersect
            # Where solid removed object was AND physics changed in that same pixel
            overlap_mask = black_mask & grey_mask

            # Step 4: Pure regions (no overlap)
            pure_black = black_mask & ~grey_mask
            pure_grey = grey_mask & ~black_mask

            # Assign values: White=255 (default), Black=0, Grey=127, Overlap=63
            m[pure_black] = 0      # Solid removed objects only (no physics change in that pixel)
            m[pure_grey] = 127     # Physics changes only (shadows, motion, etc.)
            m[overlap_mask] = 63   # Both (solid object was here AND physics changed here)

            iio.imwrite(tmpMask / Path(pA).name.replace("rgba_", "mask_"), np.stack([m,m,m], axis=-1))

        # Encode videos
        fps = int(FLAGS_like.frame_rate)
        encode_sequence(str(tmpA / "rgba_*.png"), out_dir / "rgb_full.mp4", fps)
        encode_sequence(str(tmpC / "rgba_*.png"), out_dir / "rgb_removed_objects_invisible.mp4", fps)
        encode_sequence(str(tmpB / "rgba_*.png"), out_dir / "rgb_altered_physics.mp4", fps)
        encode_sequence(str(tmpMask / "mask_*.png"), out_dir / "mask.mp4", fps, lossless=True)

        # Save metadata - convert numpy types to Python native types
        metadata = {
            'num_objects': int(num_objects),
            'num_removed': int(num_to_remove),
            'removed_indices': [int(idx) for idx in objects_to_remove],
            'target_hit': bool(target_hit),
            'background': str(hdri_id),
            'camera_motion': str(camera_motion_type),
            'videos': {
                'rgb_full.mp4': 'Pass A: All objects visible, original physics',
                'rgb_removed_objects_invisible.mp4': 'Pass C: Removed objects invisible but physically present (same physics as A)',
                'rgb_altered_physics.mp4': 'Pass B: Removed objects completely gone, altered physics',
                'mask.mp4': 'Tri-mask showing object removal and physics changes'
            },
            'mask_values': {
                '0': 'Pure black - Solid removed object pixels only (from Pass A segmentation)',
                '63': 'Overlap - Solid removed object was here AND physics changed here',
                '127': 'Pure grey - Physics changes only: motion, shadows of removed objects, etc. (C-B RGB diff)',
                '255': 'White - Background/unchanged'
            },
            'mask_encoding': 'Lossless H.264 (qp=0) codec to preserve discrete values'
        }
        import json
        with open(out_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[DEBUG] ✓ Scene complete: {num_objects} objects, {num_to_remove} removed (3-pass render)")

    finally:
        for p in (tmpA, tmpB, tmpC, tmpMask):
            if p and p.exists():
                shutil.rmtree(p, ignore_errors=True)
        if os.path.isdir(tmp_scratch):
            shutil.rmtree(tmp_scratch, ignore_errors=True)
        cleanup_scene()
        gc.collect()

def main():
    parser = kb.ArgumentParser()
    parser.add_argument("--objects_split", choices=["train","test"], default="train")
    parser.add_argument("--backgrounds_split", choices=["train","test"], default="train")
    parser.add_argument("--kubasic_assets", type=str, default="gs://kubric-public/assets/KuBasic/KuBasic.json")
    parser.add_argument("--gso_assets", type=str, default="gs://kubric-public/assets/GSO/GSO.json")
    parser.add_argument("--hdri_assets", type=str, default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    parser.set_defaults(frame_start=1, frame_end=60, frame_rate=12, resolution=384, step_rate=240)
    parser.add_argument("--out_prefix", type=str, default="variable_obj_3")
    parser.add_argument("--num_pairs", type=int, default=200)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--fast", action="store_true")

    FLAGS = parser.parse_args()

    if FLAGS.fast:
        FLAGS.frame_end = 24
        FLAGS.frame_rate = 8
        spp = 16
    else:
        spp = DEFAULT_SPP

    # Load assets
    print("[MAIN] Loading assets...")
    kubasic_source = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    gso_source = kb.AssetSource.from_manifest(FLAGS.gso_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

    _, rng, output_dir, _ = kb.setup(FLAGS)
    root = output_dir / FLAGS.out_prefix
    root.mkdir(parents=True, exist_ok=True)

    print(f"[MAIN] Generating {FLAGS.num_pairs} video pairs with variable objects...")
    successful = 0
    failed = 0

    for i in range(FLAGS.start_index, FLAGS.start_index + FLAGS.num_pairs):
        out_dir = root / f"{i:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        local_rng = np.random.RandomState(i + 2001)
        print(f"\n[BATCH {i:05d}/{FLAGS.start_index + FLAGS.num_pairs - 1:05d}] Starting...")

        try:
            generate_one(local_rng, out_dir,
                        kubasic_source, gso_source, hdri_source,
                        objects_split=FLAGS.objects_split,
                        backgrounds_split=FLAGS.backgrounds_split,
                        frame_start=FLAGS.frame_start,
                        frame_end=FLAGS.frame_end,
                        frame_rate=FLAGS.frame_rate,
                        resolution=FLAGS.resolution,
                        step_rate=FLAGS.step_rate,
                        spp=spp)
            print(f"[BATCH {i:05d}] ✓ Success")
            successful += 1
        except Exception as e:
            print(f"[BATCH {i:05d}] ✗ ERROR: {e}")
            failed += 1
            cleanup_scene()
            gc.collect()

    print(f"\n[MAIN] ===== BATCH COMPLETE =====")
    print(f"[MAIN] Successful: {successful}/{FLAGS.num_pairs}")
    print(f"[MAIN] Failed: {failed}/{FLAGS.num_pairs}")
    print(f"[MAIN] Output directory: {root}")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    main()