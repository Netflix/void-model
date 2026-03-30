"""
Clear the root scale of the armature and its child meshes.
Usage:
```bash
python clear_human_scale.py -d <fbx_folder_path> -o <output_folder> -g -b
```
"""
import os
from blender_helper import *
import bpy
from mathutils import Vector
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, required=True,
                    help="The folder containing the FBX file to process")
parser.add_argument("-o", "--output_folder", type=str, default='',
                    help="The folder to save the processed FBX file, if not specified, the processed FBX file will be saved in the same folder as the original FBX file")
parser.add_argument("-g", "--glb", action='store_true',
                    help="Whether to save the GLB file")
parser.add_argument("-b", "--blend", action='store_true',
                    help="Whether to save the BLEND file")
args = parser.parse_args()

fbx_folder_path = args.dir
output_folder = args.output_folder
folder = fbx_folder_path.split('/')[-1]
print(f"Processing humoto sequence: {fbx_folder_path}")

def bake_scale_into_animation(obj):
    """
    Applies the object-level scale of a selected armature and its child meshes
    by baking a new, corrected animation clip.

    This script performs a true bake by iterating through every frame:
    1. Creates a new, empty animation action to store the baked result.
    2. Applies the object-level scale to the armature and child meshes.
    3. Steps through each frame of the original animation.
    4. On each frame, it recalculates the bone locations based on the original
       armature scale and inserts a new keyframe into the new action.
    5. Finally, it assigns the new, baked action to the armature.
    """
    
    # --- 1. SETUP AND VALIDATION ---
    context = bpy.context
    
    if not obj or obj.type != 'ARMATURE':
        print("Error: Please select an Armature object first.")
        return {'CANCELLED'}

    if context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    armature = obj
    original_scale = armature.scale.copy()
    
    if original_scale.x == 1.0 and original_scale.y == 1.0 and original_scale.z == 1.0:
        print("Info: Armature scale is already (1, 1, 1). No action needed.")
        return {'FINISHED'}

    if not armature.animation_data or not armature.animation_data.action:
        print("Error: Armature has no active action to bake. Please assign an animation.")
        return {'CANCELLED'}

    print("Starting animation bake process...")

    original_action = armature.animation_data.action
    pose_bones = armature.pose.bones

    # --- 2. CREATE A NEW, EMPTY ACTION FOR THE BAKED DATA ---
    new_action_name = f"{original_action.name}_Baked"
    new_action = bpy.data.actions.new(name=new_action_name)
    print(f"Created new action for baking: '{new_action_name}'")

    # --- 3. APPLY SCALE TO ARMATEUR AND CHILD MESHES ---
    print("Applying object-level scale to the armature and its children...")
    
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    context.view_layer.objects.active = armature
    
    for child in armature.children:
        if child.type == 'MESH':
            child.select_set(True)
            
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    print("Armature and child mesh scales have been applied.")

    # --- 4. BAKE THE ANIMATION FRAME BY FRAME ---
    frame_start, frame_end = int(original_action.frame_range[0]), int(original_action.frame_range[1])
    print(f"Baking from frame {frame_start} to {frame_end}...")

    # Temporarily assign the original action to read from it
    armature.animation_data.action = original_action
    
    for frame in range(frame_start, frame_end + 1):
        context.scene.frame_set(frame)
        
        for bone in pose_bones:
            # Calculate the new location based on the original object scale
            new_location = bone.location * original_scale
            
            # --- Insert keyframes into the NEW action ---
            # Location
            fc_loc_x = new_action.fcurves.find(f'pose.bones["{bone.name}"].location', index=0)
            if not fc_loc_x: fc_loc_x = new_action.fcurves.new(f'pose.bones["{bone.name}"].location', index=0)
            fc_loc_x.keyframe_points.insert(frame, new_location.x)
            
            fc_loc_y = new_action.fcurves.find(f'pose.bones["{bone.name}"].location', index=1)
            if not fc_loc_y: fc_loc_y = new_action.fcurves.new(f'pose.bones["{bone.name}"].location', index=1)
            fc_loc_y.keyframe_points.insert(frame, new_location.y)
            
            fc_loc_z = new_action.fcurves.find(f'pose.bones["{bone.name}"].location', index=2)
            if not fc_loc_z: fc_loc_z = new_action.fcurves.new(f'pose.bones["{bone.name}"].location', index=2)
            fc_loc_z.keyframe_points.insert(frame, new_location.z)
            
            # Rotation (unmodified, just copied over)
            fc_rot_w = new_action.fcurves.find(f'pose.bones["{bone.name}"].rotation_quaternion', index=0)
            if not fc_rot_w: fc_rot_w = new_action.fcurves.new(f'pose.bones["{bone.name}"].rotation_quaternion', index=0)
            fc_rot_w.keyframe_points.insert(frame, bone.rotation_quaternion.w)

            fc_rot_x = new_action.fcurves.find(f'pose.bones["{bone.name}"].rotation_quaternion', index=1)
            if not fc_rot_x: fc_rot_x = new_action.fcurves.new(f'pose.bones["{bone.name}"].rotation_quaternion', index=1)
            fc_rot_x.keyframe_points.insert(frame, bone.rotation_quaternion.x)

            fc_rot_y = new_action.fcurves.find(f'pose.bones["{bone.name}"].rotation_quaternion', index=2)
            if not fc_rot_y: fc_rot_y = new_action.fcurves.new(f'pose.bones["{bone.name}"].rotation_quaternion', index=2)
            fc_rot_y.keyframe_points.insert(frame, bone.rotation_quaternion.y)

            fc_rot_z = new_action.fcurves.find(f'pose.bones["{bone.name}"].rotation_quaternion', index=3)
            if not fc_rot_z: fc_rot_z = new_action.fcurves.new(f'pose.bones["{bone.name}"].rotation_quaternion', index=3)
            fc_rot_z.keyframe_points.insert(frame, bone.rotation_quaternion.z)

    # --- 5. ASSIGN THE NEWLY BAKED ACTION ---
    armature.animation_data.action = new_action
    print("Baking complete. Assigned new action to armature.")

    # --- 6. CLEANUP ---
    context.scene.frame_set(frame_start)
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    armature.scale = Vector((1,1,1))
    context.view_layer.objects.active = armature

    print("\nProcess successfully completed!")
    return {'FINISHED'}


fbx_path = os.path.join(fbx_folder_path, f"{folder}.fbx")
clean_scene()
load_humoto_fbx(fbx_path)
scene = bpy.context.scene
frame_start = scene.frame_start
frame_end = scene.frame_end

armature = get_armature()
bake_scale_into_animation(armature)

if not output_folder or output_folder == '':
    output_file_name = folder + '#scale_cleared'
    base_output_path = fbx_folder_path
else:
    output_file_name = folder
    base_output_path = os.path.join(output_folder, folder)

os.makedirs(base_output_path, exist_ok=True)

fbx_output_path = os.path.join(base_output_path, f"{output_file_name}.fbx")
save_fbx(fbx_output_path)

if args.blend:
    blender_output_path = os.path.join(base_output_path, f"{output_file_name}.blend")
    print(f"Saving Blender file to: {blender_output_path}")
    save_blend(blender_output_path)

if args.glb:
    glb_output_path = os.path.join(base_output_path, f"{output_file_name}.glb")
    save_glb(glb_output_path)
