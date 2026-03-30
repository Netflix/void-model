import bpy
import os
import mathutils

def clean_scene():
    """Clean the scene by reading the home file.
    
    Returns:
        None
    """
    bpy.ops.wm.read_homefile(use_empty=True)
            
def load_humoto_fbx(fbx_path):
    """Load FBX file and return status.
    
    Args:
        fbx_path: The path to the FBX file to load.
        
    Returns:
        True if the FBX file was loaded successfully, False otherwise.
    """
    if not os.path.exists(fbx_path):
        raise FileNotFoundError(f"FBX file not found: {fbx_path}")
    
    try:
        # Import FBX
        bpy.ops.import_scene.fbx(
            filepath=fbx_path,
            use_anim=True,  # Import animation
            automatic_bone_orientation=False,  # Keep original bone orientation
            ignore_leaf_bones=False,  # Import all bones
            force_connect_children=False,  # Keep original bone connections
            anim_offset=0,
        )
        
        scene = bpy.context.scene
        max_frame = 0
        # Get animation details
        for action in bpy.data.actions:
            for fcurve in action.fcurves:
                for keyframe_point in fcurve.keyframe_points:
                    if keyframe_point.co.x > max_frame:
                        max_frame = keyframe_point.co.x
        frame_end = int(max_frame)
        frame_start = 1
            
        # Set frame range from animation
        scene.frame_start = frame_start
        scene.frame_end = frame_end
        scene.render.fps = 30
        print(f"Successfully loaded FBX. Frame range: {frame_start} - {frame_end}")
        return True
        
    except Exception as e:
        print(f"Error loading FBX: {str(e)}")
        return False
    
def get_armature(objects=None):
    """Get the armature object from the scene.
    
    Args:
        objects: The objects to search for the armature. If None, all objects in the scene are searched.
        
    Returns:
        The armature object if found, None otherwise.
    """
    if objects is None:
        objects = bpy.data.objects
    armature = None
    for obj in objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    return armature

def get_all_objects():
    """Get all objects in the scene.
    
    Returns:
        A list of all objects in the scene.
    """
    objects = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if obj.parent is not None and 'mixamo' in obj.name:
            continue
        if obj.type == 'MESH':
            objects.append(obj)
    objects.sort(key=lambda obj: obj.name)
    return objects
    
def reset_base_scale_animation(obj):
    """Reset the base scale of an object and its animation.
    
    Args:
        obj: The object to reset the base scale of.
        
    Returns:
        None
    """
    scene = bpy.context.scene
    start_frame = scene.frame_start
    end_frame = scene.frame_end
    scene.frame_set(start_frame)
    
    target_base_scale = mathutils.Vector((1, 1, 1))
    current_base_scale = obj.scale.copy()
    
    scale_factor = mathutils.Vector([t/c if c != 0 else 1.0
                                     for t, c in zip(target_base_scale, current_base_scale)])
    mesh = obj.data
    if mesh and hasattr(mesh, 'vertices'):
        for v in mesh.vertices:
            v.co.x /= scale_factor[0]
            v.co.y /= scale_factor[1]
            v.co.z /= scale_factor[2]
        mesh.update()
        
    keyframed_scales = {}
    has_scale_keyframes = False
    
    if obj.animation_data and obj.animation_data.action:
        action = obj.animation_data.action
        for fcurve in action.fcurves:
            if fcurve.data_path == 'scale':
                has_scale_keyframes = True
                for frame in range(start_frame, end_frame + 1):
                    scene.frame_set(frame)
                    if frame not in keyframed_scales:
                        keyframed_scales[frame] = obj.scale.copy()
    
    if not has_scale_keyframes:
        for frame in range(start_frame, end_frame + 1):
            keyframed_scales[frame] = obj.scale.copy()
        
    obj.scale = target_base_scale
    
    for frame, original_scale in keyframed_scales.items():
        scene.frame_set(frame)
        new_scale = mathutils.Vector([original_scale[i] * scale_factor[i] for i in range(3)])
        obj.scale = new_scale
        obj.keyframe_insert(data_path="scale", frame=frame)
        
def save_fbx(fbx_output_path):
    """Save the scene as an FBX file.
    
    Args:
        fbx_output_path: The path to save the FBX file.
        
    Returns:
        None
    """
    bpy.ops.export_scene.fbx(
        filepath=fbx_output_path,
        apply_scale_options='FBX_SCALE_ALL',
        bake_space_transform=False,
        object_types={'ARMATURE', 'MESH'},
        use_mesh_modifiers=True,
        bake_anim=True,  # Enable animation baking
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,  # No simplification
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )
    
def save_glb(glb_output_path):
    """Save the scene as a GLB file.
    
    Args:
        glb_output_path: The path to save the GLB file.
        
    Returns:
        None
    """
    bpy.ops.export_scene.gltf(
        filepath=glb_output_path,
        export_format='GLB',
        export_apply=True,
        export_animations=True,
        export_frame_range=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_colors=True,
        export_cameras=False,
        export_lights=False,
        export_extras=False,
        export_image_format='AUTO',
        will_save_settings=False,
    )
    
def save_blend(blend_output_path):
    """Save the scene as a BLEND file.
    
    Args:
        blend_output_path: The path to save the BLEND file.
        
    Returns:
        None
    """
    # Save the Blender file
    if os.path.exists(blend_output_path):
        os.remove(blend_output_path)
    bpy.ops.wm.save_as_mainfile(filepath=blend_output_path)
    
