# HuMoTo Counterfactual Video Generation Pipeline

This repository provides the code to generate **counterfactual paired videos** from the [HuMoTo](https://4d-humans.github.io/) human motion capture dataset. For each scene, the pipeline produces:

- **`rgb_full.mp4`** — Full scene with human + objects + textures
- **`rgb_removed.mp4`** — Same scene with human removed (objects fall via physics)
- **`mask.mp4`** — Quad-mask encoding what changed (4 values: keep/modify/overlap/remove)
- **`metadata.json`** — Scene info, textures used, character, camera variant

## Prerequisites

### 1. HuMoTo Dataset (Required)

You must request access to the HuMoTo dataset directly from the authors at [adobe-research/humoto](https://github.com/adobe-research/humoto). Once approved, download the dataset:

```bash
# After obtaining access, download to ./humoto_release/humoto_0805/
aws s3 cp s3://g3i-data/4Danimation/humoto/licensed_release/release_with_research_license/ ./humoto_release/ --recursive
```

### 2. Blender (Required)

Install [Blender](https://www.blender.org/download/) (tested with Blender 3.x/4.x). The rendering scripts run via:
```bash
blender --background --python <script.py> -- [args]
```

### 3. Character Models (Required)

Download Remy and Sophie from [Mixamo](https://www.mixamo.com/) (free Adobe account required) and place them at:
```
human_model/Remy_mixamo_bone.fbx
human_model/Sophie_mixamo_bone.fbx
```

### 4. PBR Textures (Optional but recommended)

Download PBR texture packs of your choice (e.g., from [ambientCG](https://ambientcg.com/) or [Poly Haven](https://polyhaven.com/)) and organize them into directories. Pass texture directories to the renderer via `--floor_texture`, `--wall_texture`, and `--object_texture` flags.

### 5. Python Dependencies

```bash
pip install numpy Pillow tqdm pyyaml streamlit
```

---

## Pipeline

### Step 1: Character Conversion

Convert HuMoTo sequences to use Remy/Sophie character models. This is a 4-step process per sequence: clear scale, transfer character skeleton, extract pose data, copy metadata.

```bash
# Edit paths in the script first (HUMOTO_DIR, FBX paths, etc.)
bash convert_split_remy_sophie.sh
```

Or use the Python version for more control:
```bash
python convert_all_scenarios_for_characters.py \
    --humoto_dir ./humoto_release/humoto_0805 \
    --output_dir ./humoto_characters_converted \
    --remy_fbx ./human_model/Remy_mixamo_bone.fbx \
    --sophie_fbx ./human_model/Sophie_mixamo_bone.fbx
```

**Output:** `humoto_characters_converted/{remy,sophie}/<sequence>/` with `.yaml`, `.fbx`, `.pkl` files.

### Step 2: Physics Configuration

The included `physics_config.json` contains our manual per-sequence settings specifying which objects are static (e.g., table, shelf) vs. which fall when the human is removed (e.g., mug, plate).

If you want to review or modify these settings:

```bash
# Generate preview frames for review
python generate_review_frames.py --data_dir ./humoto_characters_converted

# Launch the interactive review UI
streamlit run physics_review_ui.py
```

Or generate a fresh config template:
```bash
python generate_physics_config.py --data_dir ./humoto_release/humoto_0805
```

### Step 3: Render Paired Videos

The main rendering script produces 3 passes per sequence (rgb_full, rgb_removed, silhouette) and generates the quad-mask.

```bash
blender --background --python render_paired_videos_blender_quadmask.py -- \
    -d ./humoto_release/humoto_0805 \
    -o ./output \
    -s <sequence_name> \
    -m ./humoto_release/humoto_0805 \
    --use_characters \
    --characters_dir ./humoto_characters_converted \
    --enable_physics \
    --add_walls \
    --floor_texture /path/to/floor/textures \
    --wall_texture /path/to/wall/textures \
    --object_texture /path/to/object/textures \
    --seed 42
```

**Key flags:**
| Flag | Description |
|------|-------------|
| `-d` | Path to HuMoTo dataset directory |
| `-o` | Output directory |
| `-s` | Sequence name(s) to render |
| `-m` | Object model directory |
| `--use_characters` | Use Remy/Sophie instead of solid-color human |
| `--force_character` | Force `remy` or `sophie` |
| `--enable_physics` | Enable rigid body physics for removed-human pass |
| `--add_walls` | Add room walls to scene |
| `--camera_variant` | Camera trajectory variant (`v1`-`v9`) |
| `--resolution_x/y` | Render resolution (default: 1600x900) |
| `--fps` | Frame rate (default: 12) |
| `--target_frames` | Number of frames to render (default: 60) |

### Step 4: Post-Processing (Optional)

Convert tri-masks to quad-masks (if not already generated):
```bash
python convert_trimask_to_quadmask.py --input_dir ./output --output_dir ./output_quadmask
```

Convert to grid-aligned hybrid masks:
```bash
python convert_masks_to_grid_hybrid.py --input_dir ./output --output_dir ./output_grid
```

---

## Kubric Pipeline (Object-Only Counterfactuals)

A separate, self-contained pipeline using [Kubric](https://github.com/google-research/kubric) for generating counterfactual videos with Google Scanned Objects (GSO). Unlike the HuMoTo pipeline above, this does **not** involve human characters — it generates scenes with 3-7 objects where some are removed to observe changes in physics dynamics.

### How it works

The script runs **3 rendering passes** per scene:

1. **Pass A (`rgb_full.mp4`)** — All objects present, full physics simulation. Some objects are launched toward a target.
2. **Pass C (`rgb_removed_objects_invisible.mp4`)** — Same physics as Pass A, but removed objects are hidden from the renderer (their physical influence remains).
3. **Pass B (`rgb_altered_physics.mp4`)** — Removed objects are completely absent. Physics re-simulated without them, so the target object behaves differently.

The quad-mask is computed from all 3 passes:
- **Pass A segmentation** identifies removed-object pixels (black/0)
- **Pass C vs B diff** identifies physics changes (grey/127)
- **Overlap** of both (dark grey/63)

### Usage

```bash
python kubric_variable_objects.py \
    --num_pairs 200 \
    --resolution 384 \
    --frame_end 60 \
    --frame_rate 12 \
    --out_prefix my_dataset

# Fast mode (fewer frames, lower quality)
python kubric_variable_objects.py --num_pairs 10 --fast
```

**Key flags:**
| Flag | Description |
|------|-------------|
| `--num_pairs` | Number of video pairs to generate (default: 200) |
| `--start_index` | Starting index for batch generation |
| `--resolution` | Render resolution (default: 384) |
| `--frame_end` | Number of frames (default: 60) |
| `--frame_rate` | FPS (default: 12) |
| `--fast` | Fewer frames (24) and lower samples for quick testing |
| `--kubasic_assets` | KuBasic asset manifest (default: GCS public bucket) |
| `--gso_assets` | GSO asset manifest (default: GCS public bucket) |
| `--hdri_assets` | HDRI background manifest (default: GCS public bucket) |

**Output per scene:**
```
00000/
├── rgb_full.mp4                       # All objects, original physics
├── rgb_removed_objects_invisible.mp4   # Removed objects hidden, same physics
├── rgb_altered_physics.mp4             # Removed objects gone, re-simulated physics
├── mask.mp4                            # Quad-mask (lossless)
└── metadata.json                       # Object count, which removed, collisions, etc.
```

### Dependencies

Kubric requires its own environment. See [Kubric installation](https://github.com/google-research/kubric#installation). Assets (KuBasic, GSO, HDRI Haven) are fetched automatically from Google Cloud Storage.

```bash
pip install kubric pybullet imageio imageio-ffmpeg
```

---

## Quad-Mask Format

The quad-mask encodes 4 pixel-level labels:

| Value | Meaning |
|-------|---------|
| **0** (black) | Human was here, nothing moved underneath (pure removal) |
| **63** (dark grey) | Human was here AND an object moved underneath (overlap) |
| **127** (grey) | Object moved / scene changed (not under human) |
| **255** (white) | No change (keep as-is) |

---

## File Structure

```
humoto_proc_release/
├── render_paired_videos_blender_quadmask.py  # Main 3-pass renderer
├── blender_utils.py                          # Blender utility functions
├── object_texture_mapping.py                 # Object-to-texture category mapping
├── convert_split_remy_sophie.sh              # Character conversion (bash)
├── convert_all_scenarios_for_characters.py   # Character conversion (python)
├── scripts/
│   ├── clear_human_scale.py                  # Conversion step 1
│   ├── transfer_human_model.py               # Conversion step 2
│   ├── extract_pk_data.py                    # Conversion step 3
│   └── blender_helper.py                     # Shared Blender helpers
├── human_model/
│   ├── bone_names.py                         # Mixamo bone name definitions
│   ├── human_model.py                        # Human model utilities
│   └── *.json                                # Bone structure definitions
├── physics_config.json                       # Pre-configured physics settings
├── apply_physics_config.py                   # Load/query physics config
├── generate_physics_config.py                # Generate config template
├── physics_review_ui.py                      # Streamlit UI for physics review
├── generate_review_frames.py                 # Generate preview frames
├── generate_review_frames_direct.py          # Direct bpy version
├── convert_trimask_to_quadmask.py            # Tri-mask → quad-mask conversion
├── convert_masks_to_grid_hybrid.py           # Grid-aligned mask conversion
└── kubric_variable_objects.py                # Kubric object counterfactual generator
```

---

## Citation

If you use this pipeline, please cite the HuMoTo dataset according to their license terms.
