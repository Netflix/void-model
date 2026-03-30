"""
Extract the skeleton and object data from the FBX file.
Usage:
```bash
python extract_pk_data.py -d <fbx_folder_path> -o <output_folder>
```
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy
import numpy as np
import pickle
from human_model.bone_names import MIXAMO_BONE_NAMES
from blender_helper import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, required=True,
                    help="The folder containing the FBX file to process")
parser.add_argument("-o", "--output_folder", type=str, default='',
                    help="The folder to save the processed FBX file, if not specified, the processed FBX file will be saved in the same folder as the original FBX file")
parser.add_argument("-m", "--object_model", action='store_true',
                    help="Whether to extract the object model data")
args = parser.parse_args()

fbx_folder_path = args.dir
output_folder = args.output_folder
folder = fbx_folder_path.split('/')[-1]

def extract_skeleton_data():
    armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    if armature is None:
        raise Exception("No armature found")
    frame_data = []
    scene = bpy.context.scene
    start_frame = scene.frame_start
    end_frame = scene.frame_end
    scale = list(armature.scale)
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        cur_frame = {}
        for bone in armature.pose.bones:
            location = list(bone.location)
            location = [l * s for l,s in zip(location, scale)]
            if bone.rotation_mode == 'QUATERNION':
                rotation = list(bone.rotation_quaternion)
            else:
                quat = bone.rotation_euler.to_quaternion()
                rotation = list(quat)
            cur_frame[bone.name] = rotation + location
        frame_data.append(cur_frame)
    return frame_data

def extract_object_data():
    object_data = {}
    objects = get_all_objects()
    scene = bpy.context.scene
    for obj in objects:
        scene.frame_set(scene.frame_start)
        object_data[obj.name] = []
        start_frame = scene.frame_start
        end_frame = scene.frame_end
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)
            location = list(obj.location)
            if obj.rotation_mode == 'QUATERNION':
                rotation = list(obj.rotation_quaternion)
            else:
                quat = obj.rotation_euler.to_quaternion()
                rotation = list(quat)
            object_data[obj.name].append(rotation + location)
    return object_data

def extract_object_model_data():
    """
    Extract mesh vertices and faces from objects.
    Works in both object mode and edit mode.
    """
    object_data = {}
    objects = get_all_objects()
    scene = bpy.context.scene
    
    for obj in objects:
        scene.frame_set(scene.frame_start)
        
        # Only process mesh objects
        if obj.type != 'MESH':
            continue
        
        # Select the object and make it active
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # If in object mode, access mesh data directly
        mesh = obj.data
        
        # Extract vertices (in local coordinates)
        vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
        
        # Extract faces (vertex indices)
        faces = []
        for p in mesh.polygons:
            for iters in range(len(p.vertices) - 2):
                faces.append([p.vertices[0], p.vertices[iters + 1], p.vertices[iters + 2]])
        faces = np.array(faces, dtype=np.int64)
        
        object_data[obj.name] = {
            'mesh': (vertices, faces),
        }
    
    return object_data

fbx_path = os.path.join(fbx_folder_path, f'{folder}.fbx')
assert os.path.exists(fbx_path), f"FBX file not found: {fbx_path}"
assert fbx_path.endswith('.fbx'), f"Invalid file format: {fbx_path}"

if not output_folder or output_folder == '':
    save_path = fbx_folder_path
else:
    save_path = os.path.join(output_folder, folder)
os.makedirs(save_path, exist_ok=True)

clean_scene()
load_humoto_fbx(fbx_path)

skeleton_data = extract_skeleton_data()
object_data = extract_object_data()
data = {'armature': skeleton_data, 
        'objects': object_data}

# Extract object model data (vertices and faces) if requested
if args.object_model:
    object_model_data = extract_object_model_data()
    data['object_models'] = object_model_data

# data structure:
# armature [
#     {
#         'BONE_NAME': [quat, loc]
#     }
# ]
# objects
#     {
#         'OBJECT_NAME': [[quat, loc]]
#     }
# object_models (when --object_model flag is used)
#     {
#         'OBJECT_NAME': {
#             'vertices': np.array (N, 3),
#             'faces': np.array (M, P) where P is number of vertices per face,
#         }
#     }

save_data_path = os.path.join(save_path, f'{folder}.pkl')
with (open(save_data_path, "wb")) as f:
    pickle.dump(data, f)
    
