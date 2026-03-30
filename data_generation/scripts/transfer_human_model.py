"""
Transfer human model to the target armature.

Usage:
```bash
python transfer_human_model.py -d <fbx_folder_path> -m <human_model_path> -o <output_folder> -g -b
```
"""
import bpy
import os
from blender_helper import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, required=True,
                    help="The folder containing the FBX file to process")
parser.add_argument("-m", "--human_model", type=str, required=True,
                    help="The path to the human model FBX file")
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

class ArmatureAnimationTransfer:
    """
    A class to transfer animation from a source armature to a target armature,
    accounting for differences in bone orientation.
    """
    def __init__(self):
        pass

    def transfer_animation(self, source_armature, target_armature, bone_mapping=None, frame_range=None):
        """
        Transfers animation from the source to the target 
        armature over a specified frame range.
        
        Args:
            source_armature: The armature object to copy animation from.
            target_armature: The armature object to copy animation to.
            bone_mapping: A dictionary mapping source bone names to target bone names. 
                         If None, it assumes bone names are identical.
            frame_range: A tuple (start_frame, end_frame). If None, uses the scene's frame range.
        """
        if bone_mapping is None:
            bone_mapping = {bone.name: bone.name for bone in source_armature.data.bones}

        if frame_range is None:
            frame_range = (bpy.context.scene.frame_start, bpy.context.scene.frame_end)

        # Ensure the target armature has an animation data block
        if not target_armature.animation_data:
            target_armature.animation_data_create()
        if not target_armature.animation_data.action:
            target_armature.animation_data.action = bpy.data.actions.new(name=f"{target_armature.name}_Action")

        # Pre-calculate bone difference matrices for efficiency
        bone_diff_matrices = {}
        for source_bone_name, target_bone_name in bone_mapping.items():
            if source_bone_name in source_armature.data.bones and target_bone_name in target_armature.data.bones:
                source_edit_bone = source_armature.data.bones[source_bone_name]
                target_edit_bone = target_armature.data.bones[target_bone_name]
                
                source_bone_matrix = source_edit_bone.matrix_local
                target_bone_matrix = target_edit_bone.matrix_local
                
                bone_diff_matrices[source_bone_name] = target_bone_matrix.inverted() @ source_bone_matrix

        # Iterate through each frame and transfer the pose
        for frame in range(frame_range[0], frame_range[1] + 1):
            bpy.context.scene.frame_set(frame)
            
            for source_bone_name, target_bone_name in bone_mapping.items():
                if source_bone_name in source_armature.pose.bones and target_bone_name in target_armature.pose.bones:
                    source_pose_bone = source_armature.pose.bones[source_bone_name]
                    target_pose_bone = target_armature.pose.bones[target_bone_name]
                    
                    bone_diff_matrix = bone_diff_matrices.get(source_bone_name)
                    if not bone_diff_matrix:
                        continue

                    # --- Location Transfer ---
                    corrected_translation = bone_diff_matrix.to_3x3() @ source_pose_bone.location
                    target_pose_bone.location = corrected_translation
                    target_pose_bone.keyframe_insert(data_path="location", frame=frame)

                    # --- Rotation Transfer ---
                    if source_pose_bone.rotation_mode == 'QUATERNION':
                        source_rotation_matrix = source_pose_bone.rotation_quaternion.to_matrix().to_4x4()
                    else: # EULER
                        source_rotation_matrix = source_pose_bone.rotation_euler.to_matrix().to_4x4()
                    
                    corrected_rotation_matrix = bone_diff_matrix @ source_rotation_matrix @ bone_diff_matrix.inverted()
                    
                    if target_pose_bone.rotation_mode == 'QUATERNION':
                        target_pose_bone.rotation_quaternion = corrected_rotation_matrix.to_quaternion()
                        target_pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
                    else: # EULER
                        target_pose_bone.rotation_euler = corrected_rotation_matrix.to_euler(target_pose_bone.rotation_mode)
                        target_pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)

                    # --- Scale Transfer ---
                    target_pose_bone.scale = source_pose_bone.scale
                    target_pose_bone.keyframe_insert(data_path="scale", frame=frame)

        print(f"Animation transferred from {source_armature.name} to {target_armature.name}")

def transfer_animation_between_armatures(source_name, target_name, bone_mapping=None, frame_range=None):
    """A helper function to easily transfer animation between two armatures in the scene.

    Args:
        source_name: The name of the source armature object.
        target_name: The name of the target armature object.
        bone_mapping: A dictionary mapping source bone names to target bone names.
        frame_range: A tuple (start_frame, end_frame) for the animation transfer.
    """
    source_armature = bpy.data.objects.get(source_name)
    target_armature = bpy.data.objects.get(target_name)

    if not source_armature or not target_armature:
        print(f"Error: Could not find armatures '{source_name}' or '{target_name}'")
        return False

    if source_armature.type != 'ARMATURE' or target_armature.type != 'ARMATURE':
        print("Error: Both objects must be armatures.")
        return False

    # Set the target armature to POSE mode
    original_active = bpy.context.view_layer.objects.active
    original_mode = bpy.context.mode
    
    bpy.context.view_layer.objects.active = target_armature
    if bpy.context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')

    # Perform the animation transfer
    animation_transfer = ArmatureAnimationTransfer()
    animation_transfer.transfer_animation(source_armature, target_armature, bone_mapping, frame_range)

    # Restore the original context
    if original_mode != 'POSE':
        bpy.ops.object.mode_set(mode=original_mode)
    if original_active:
        bpy.context.view_layer.objects.active = original_active
        
    return True

clean_scene()
fbx_path = os.path.join(fbx_folder_path, f"{folder}.fbx")
load_humoto_fbx(fbx_path)

scene = bpy.context.scene
frame_start = scene.frame_start
frame_end = scene.frame_end
armature = get_armature()

human_model_path = args.human_model

bpy.ops.import_scene.fbx(
    filepath=human_model_path,
    use_anim=False,
    automatic_bone_orientation=False,
    ignore_leaf_bones=False,
    force_connect_children=False,
    anim_offset=0,
)

imported = bpy.context.selected_objects
target_human_model = get_armature(imported)
    
transfer_animation_between_armatures(armature.name, target_human_model.name, frame_range=(frame_start, frame_end))

source_armature_name = armature.name

# remove armature and its children
for child in armature.children:
    bpy.data.objects.remove(child)
bpy.data.objects.remove(armature)

target_human_model.name = source_armature_name

if not output_folder or output_folder == '':
    save_path = fbx_folder_path
    human_model_name = human_model_path.split('/')[-1].split('.')[0]
    save_file_name = folder + "#" + human_model_name
else:
    save_path = os.path.join(output_folder, folder)
    save_file_name = folder

os.makedirs(save_path, exist_ok=True)

fbx_output_path = os.path.join(save_path, f"{save_file_name}.fbx")
save_fbx(fbx_output_path)
if args.blend:
    blender_output_path = os.path.join(save_path, f"{save_file_name}.blend")
    print(f"Saving Blender file to: {blender_output_path}")
    save_blend(blender_output_path)

if args.glb:
    glb_output_path = os.path.join(save_path, f"{save_file_name}.glb")
    save_glb(glb_output_path)
