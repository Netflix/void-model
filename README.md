<div align="center">
<img src="assets/void-logo-web.png" width="195" />
</div>

# VOID: Video Object and Interaction Deletion

<div style="line-height: 1;">
  <a href="https://void-model.github.io/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website-VOID-4285F4" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2604.02296" target="_blank" style="margin: 2px;">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-VOID-FBBC06" style="display: inline-block; vertical-align: middle;"/>
  </a>
      <a href="https://huggingface.co/spaces/sam-motamed/VOID" target="_blank" style="margin: 2px;">
    <img alt="Data" src="https://img.shields.io/badge/🤗Gradio-Demo-AB8165" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/netflix/void-model" target="_blank" style="margin: 2px;">
    <img alt="Models" src="https://img.shields.io/badge/🤗%20HuggingFace-Models-orange" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/papers/2604.02296" target="_blank" style="margin: 2px;">
    <img alt="Models" src="https://img.shields.io/badge/🤗%20HuggingFace-Paper-yellow" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://colab.research.google.com/github/netflix/void-model/blob/main/notebook.ipynb" target="_blank" style="margin: 2px;">
    <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<h4>

[Saman Motamed](https://sam-motamed.github.io/)<sup>1,2</sup>,
[William Harvey](https://scholar.google.com/citations?user=kDd7nBkAAAAJ&hl=en)<sup>1</sup>,
[Benjamin Klein](https://scholar.google.com/citations?user=xkX9W9QAAAAJ&hl=en)<sup>1</sup>,
[Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en)<sup>2</sup>,
[Zhuoning Yuan](https://zhuoning.cc/)<sup>1</sup>,
[Ta-Ying Cheng](https://ttchengab.github.io/)<sup>1</sup>

<sup>1</sup>Netflix &nbsp;&nbsp; <sup>2</sup>INSAIT, Sofia University "St. Kliment Ohridski"

</h4>

<hr>

VOID removes objects from videos along with all interactions they induce on the scene — not just secondary effects like shadows and reflections, but physical interactions like objects falling when a person is removed. It is built on top of [CogVideoX](https://github.com/THUDM/CogVideo) and fine-tuned for video inpainting with interaction-aware mask conditioning.

> **Example:** If a person holding a guitar is removed, VOID also removes the person's effect on the guitar — causing it to fall naturally.

<video src="https://github.com/user-attachments/assets/ad174ca0-2feb-45f9-9405-83167037d9be" width="100%" controls autoplay loop muted></video>

---

## 🤖 Models

VOID uses two transformer checkpoints, trained sequentially. You can run inference with Pass 1 alone or chain both passes for higher temporal consistency.

| Model | Description | HuggingFace |
|-------|-------------|-------------|
| **VOID Pass 1** | Base inpainting model | [Download](https://huggingface.co/netflix/void-model/blob/main/void_pass1.safetensors) |
| **VOID Pass 2** | Warped-noise refinement model | [Download](https://huggingface.co/netflix/void-model/blob/main/void_pass2.safetensors) |

Place checkpoints anywhere and pass the path via `--config.video_model.transformer_path` (Pass 1) or `--model_checkpoint` (Pass 2).

---

## ▶️ Quick Start

The fastest way to try VOID is the included notebook — it handles setup, downloads the models, runs inference on a sample video, and displays the result:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/netflix/void-model/blob/main/notebook.ipynb)

> **Note:** Requires a GPU with 40GB+ VRAM (e.g., A100).

For more control over the pipeline (custom videos, Pass 2 refinement, mask generation), see the full setup and instructions below.

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Stage 1 of the mask pipeline uses Gemini via the Google AI API. Set your API key:

```bash
export GEMINI_API_KEY=your_key_here
```

Also install [SAM2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation) separately (required for mask generation):

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
```

Download the pretrained base inpainting model from HuggingFace:

```bash
hf download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP \
    --local-dir ./CogVideoX-Fun-V1.5-5b-InP
```

The inference and training scripts expect it at `./CogVideoX-Fun-V1.5-5b-InP` relative to the repo root by default.

If `ffmpeg` is not available on your system, you can use the binary bundled with `imageio-ffmpeg`:

```bash
ln -sf $(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") ~/.local/bin/ffmpeg
```

<details>
<summary><strong>📁 Expected directory structure</strong></summary>

After cloning the repo and downloading all assets, your directory should look like this:

```
VOID/
├── config/
├── datasets/
│   └── void_train_data.json
├── inference/
├── sample/                         # included sample sequences for inference
├── scripts/
├── videox_fun/
├── VLM-MASK-REASONER/
├── README.md
├── requirements.txt
│
├── CogVideoX-Fun-V1.5-5b-InP/     # hf download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP
├── void_pass1.safetensors          # download from huggingface.co/void-model (see Models above)
├── void_pass2.safetensors          # download from huggingface.co/void-model (see Models above)
├── training_data/                  # generated via data_generation/ pipeline (see Training section)
└── data_generation/                # data generation code (HUMOTO + Kubric pipelines)
```

</details>

---

## 📂 Input Format

Each video sequence lives in its own folder under a root data directory:

```
data_rootdir/
└── my-video/
    ├── input_video.mp4      # source video
    ├── quadmask_0.mp4       # quadmask (4-value mask video, see below)
    └── prompt.json          # {"bg": "background description"}
```

The `prompt.json` contains a single `"bg"` key describing the scene **after** the object has been removed — i.e. what you want the background to look like. Do not describe the object being removed; describe what remains.

```json
{ "bg": "A table with a cup on it." }         // ✅ describes the clean background
{ "bg": "A person being removed from scene." } // ❌ don't describe the removal
```

A few examples from the included samples:

| Sequence | Removed object | `bg` prompt |
|----------|---------------|-------------|
| `lime` | the glass | `"A lime falls on the table."` |
| `moving_ball` | the rubber duckie | `"A ball rolls off the table."` |
| `pillow` | the kettlebell being placed on the pillow | `"Two pillows are on the table."` |

The quadmask encodes four semantic regions per pixel:

| Value | Meaning |
|-------|---------|
| `0`   | Primary object to remove |
| `63`  | Overlap of primary + affected regions |
| `127` | Affected region (interactions: falling objects, displaced items, etc.) |
| `255` | Background (keep) |

---

## 🚀 Pipeline

<details>
<summary><strong>🎭 Stage 1 — Generate Masks</strong></summary>

The `VLM-MASK-REASONER/` pipeline generates quadmasks from raw videos using SAM2 segmentation and a VLM (Gemini) for reasoning about interaction-affected regions.

### 🖱️ Step 0 — Select points (GUI)

```bash
python VLM-MASK-REASONER/point_selector_gui.py
```

Load a JSON config listing your videos and instructions, then click on the objects to remove. Saves a `*_points.json` with the selected points.

Config format:
```json
{
  "videos": [
    {
      "video_path": "path/to/video.mp4",
      "output_dir": "path/to/output/folder",
      "instruction": "remove the person"
    }
  ]
}
```

### ⚡ Steps 1–4 — Run the full pipeline

After saving the points config, run all remaining stages automatically:

```bash
bash VLM-MASK-REASONER/run_pipeline.sh my_config_points.json
```

Optional flags:
```bash
bash VLM-MASK-REASONER/run_pipeline.sh my_config_points.json \
    --sam2-checkpoint path/to/sam2_hiera_large.pt \
    --device cuda
```

This runs the following stages in order:

| Stage | Script | Output |
|-------|--------|--------|
| 1 — SAM2 segmentation | `stage1_sam2_segmentation.py` | `black_mask.mp4` |
| 2 — VLM analysis | `stage2_vlm_analysis.py` | `vlm_analysis.json` |
| 3 — Grey mask generation | `stage3a_generate_grey_masks_v2.py` | `grey_mask.mp4` |
| 4 — Combine into quadmask | `stage4_combine_masks.py` | `quadmask_0.mp4` |

The final `quadmask_0.mp4` in each video's `output_dir` is ready to use for inference.

</details>

---

<details>
<summary><strong>🎬 Stage 2 — Inference</strong></summary>

VOID inference runs in two passes. Pass 1 is sufficient for most videos; Pass 2 adds a warped-noise refinement step for better temporal consistency on longer clips.

### ✨ Pass 1 — Base inference

```bash
python inference/cogvideox_fun/predict_v2v.py \
    --config config/quadmask_cogvideox.py \
    --config.data.data_rootdir="path/to/data_rootdir" \
    --config.experiment.run_seqs="my-video" \
    --config.experiment.save_path="path/to/output" \
    --config.video_model.model_name="path/to/CogVideoX-Fun-V1.5-5b-InP" \
    --config.video_model.transformer_path="path/to/void_pass1.safetensors"
```

To run multiple sequences at once, pass a comma-separated list:
```bash
--config.experiment.run_seqs="video1,video2,video3"
```

Key config options:

| Flag | Default | Description |
|------|---------|-------------|
| `--config.data.sample_size` | `384x672` | Output resolution (HxW) |
| `--config.data.max_video_length` | `197` | Max frames to process |
| `--config.video_model.temporal_window_size` | `85` | Temporal window for multidiffusion |
| `--config.video_model.num_inference_steps` | `50` | Denoising steps |
| `--config.video_model.guidance_scale` | `1.0` | Classifier-free guidance scale |
| `--config.system.gpu_memory_mode` | `model_cpu_offload_and_qfloat8` | Memory mode (`model_full_load`, `model_cpu_offload`, `sequential_cpu_offload`) |

The output is saved as `<save_path>/<sequence_name>.mp4`, along with a `*_tuple.mp4` side-by-side comparison.

### 🔁 Pass 2 — Warped noise refinement

Uses optical flow-warped latents from the Pass 1 output to initialize a second inference pass, improving temporal consistency.

**Single video:**
```bash
python inference/cogvideox_fun/inference_with_pass1_warped_noise.py \
    --video_name my-video \
    --data_rootdir path/to/data_rootdir \
    --pass1_dir path/to/pass1_outputs \
    --output_dir path/to/pass2_outputs \
    --model_checkpoint path/to/void_pass2.safetensors \
    --model_name path/to/CogVideoX-Fun-V1.5-5b-InP
```

**Batch:** Edit the video list and paths in `inference/pass_2_refine.sh`, then run:

```bash
bash inference/pass_2_refine.sh
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--pass1_dir` | — | Directory containing Pass 1 output videos |
| `--output_dir` | `./inference_with_warped_noise` | Where to save Pass 2 results |
| `--warped_noise_cache_dir` | `./pass1_warped_noise_cache` | Cache for precomputed warped latents |
| `--temporal_window_size` | `85` | Temporal window size |
| `--height` / `--width` | `384` / `672` | Output resolution |
| `--guidance_scale` | `6.0` | CFG scale |
| `--num_inference_steps` | `50` | Denoising steps |
| `--use_quadmask` | `True` | Use quadmask conditioning |

</details>

---

<details>
<summary><strong>✏️ Stage 3 — Manual Mask Refinement <em>(Optional)</em></strong></summary>

If the auto-generated quadmask does not accurately capture the object or its interaction region, use the included GUI editor to refine it before running inference.

```bash
python VLM-MASK-REASONER/edit_quadmask.py
```

Open a sequence folder containing `input_video.mp4` (or `rgb_full.mp4`) and `quadmask_0.mp4`. The editor shows the original video and the editable mask side by side.

**Tools:**
- **Grid Toggle** — click a grid cell to toggle the interaction region (`127` ↔ `255`)
- **Grid Black Toggle** — click a grid cell to toggle the primary object region (`0` ↔ `255`)
- **Brush (Add / Erase)** — freehand paint or erase mask regions at pixel level
- **Copy from Previous Frame** — propagate the black or grey mask from the previous frame

**Keyboard shortcuts:** `←` / `→` navigate frames, `Ctrl+Z` / `Ctrl+Y` undo/redo.

Save overwrites `quadmask_0.mp4` in place. Rerun inference from Pass 1 after saving.

</details>

---

<details>
<summary><strong>🏋️ Training</strong></summary>
  
### Training Data Generation

Due to licensing constraints on the underlying datasets, we release the **data generation code** instead of the pre-built training data. The code produces paired counterfactual videos (with/without object, plus quad-masks) from two sources:

#### Source 1: HUMOTO (Human-Object Interaction)

Generates counterfactual videos from the [HUMOTO](https://github.com/adobe-research/humoto) motion capture dataset using Blender. A human (Remy/Sophie character) interacts with objects; removing the human causes objects to fall via physics simulation.

**Prerequisites:**
1. **HUMOTO dataset** — Request access from the authors at [adobe-research/humoto](https://github.com/adobe-research/humoto). Once approved, download and place under `data_generation/humoto_release/`
2. **Blender** — Install [Blender](https://www.blender.org/download/) (tested with 3.x and 4.x). Also install `opencv-python-headless` in Blender's Python (see `data_generation/README.md`)
3. **Remy & Sophie characters** — Download from [Mixamo](https://www.mixamo.com/) (free Adobe account). Search for "Remy" and "Sophie", download each as FBX, and place at:
   ```
   data_generation/human_model/Remy_mixamo_bone.fbx
   data_generation/human_model/Sophie_mixamo_bone.fbx
   ```
4. **PBR textures** (optional) — Download texture packs from [ambientCG](https://ambientcg.com/) or [Poly Haven](https://polyhaven.com/). Without textures, objects render with realistic solid colors as fallback

**Expected directory structure after setup:**
```
data_generation/
├── humoto_release/
│   ├── humoto_0805/                    # HUMOTO sequences (.pkl, .fbx, .yaml per sequence)
│   └── humoto_objects_0805/            # Object meshes (.obj, .fbx per object)
├── human_model/
│   ├── Remy_mixamo_bone.fbx            # ← download from Mixamo
│   ├── Sophie_mixamo_bone.fbx          # ← download from Mixamo
│   ├── bone_names.py                   # included
│   └── *.json                          # included (bone structure definitions)
├── textures/                           # ← optional, user-provided PBR textures
├── physics_config.json                 # included (manual per-sequence physics settings)
├── render_paired_videos_blender_quadmask.py   # main renderer
├── convert_split_remy_sophie.sh               # character conversion script
└── ...
```

**Pipeline:**
```bash
cd data_generation

# 1. Convert HUMOTO sequences to Remy/Sophie characters
bash convert_split_remy_sophie.sh

# 2. Render paired videos (with human, without human, quad-mask)
blender --background --python render_paired_videos_blender_quadmask.py -- \
    -d ./humoto_release/humoto_0805 \
    -o ./output \
    -s <sequence_name> \
    -m ./humoto_release/humoto_objects_0805 \
    --use_characters --enable_physics --add_walls \
    --target_frames 60 --fps 12
```

A pre-configured `physics_config.json` is included specifying which objects are static vs. dynamic per sequence. See `data_generation/README.md` for full details.


#### Source 2: Kubric (Object-Only Interaction)

Generates counterfactual videos using [Kubric](https://github.com/google-research/kubric) with Google Scanned Objects. Objects are launched at a target; removing them alters the target's physics trajectory. No external dataset download required — assets are fetched from Google Cloud Storage.

```bash
cd data_generation
pip install kubric pybullet imageio imageio-ffmpeg

python kubric_variable_objects.py --num_pairs 200 --resolution 384
```

#### Training Data Format

Both pipelines output the same format expected by the training scripts:

```
training_data/
└── sequence_name/
    ├── rgb_full.mp4       # input video (with object)
    ├── rgb_removed.mp4    # target video (object removed, physics applied)
    ├── mask.mp4           # quad-mask (0/63/127/255)
    └── metadata.json
```

Point the training scripts at your generated data by updating `datasets/void_train_data.json`.

---

### Running Training

Training proceeds in two stages. Pass 1 is trained first, then Pass 2 fine-tunes from that checkpoint.

#### Pass 1 — Base inpainting model

Does not require warped noise. Trains the model to remove objects and their interactions from scratch.

```bash
bash scripts/cogvideox_fun/train_void.sh
```

Key arguments:

| Argument | Description |
|----------|-------------|
| `--pretrained_model_name_or_path` | Path to base CogVideoX inpainting model |
| `--transformer_path` | Optional starting checkpoint |
| `--train_data_meta` | Path to dataset metadata JSON |
| `--train_mode="void"` | Enables void inpainting training mode |
| `--use_quadmask` | Trains with 4-value quadmask conditioning |
| `--use_vae_mask` | Encodes mask through VAE |
| `--output_dir` | Where to save checkpoints |
| `--num_train_epochs` | Number of epochs |
| `--checkpointing_steps` | Save a checkpoint every N steps |
| `--learning_rate` | Default `1e-5` |

#### Pass 2 — Warped noise refinement model

Continues training from a Pass 1 checkpoint with optical flow-warped latent initialization, improving temporal consistency on longer videos. Requires warped noise for training data to be present.

```bash
bash scripts/cogvideox_fun/train_void_warped_noise.sh
```

Set `TRANSFORMER_PATH` to your Pass 1 checkpoint before running:

```bash
TRANSFORMER_PATH=path/to/pass1_checkpoint.safetensors bash scripts/cogvideox_fun/train_void_warped_noise.sh
```

Additional arguments specific to this stage:

| Argument | Description |
|----------|-------------|
| `--use_warped_noise` | Enables warped latent initialization during training |
| `--warped_noise_degradation` | Noise blending factor (default `0.3`) |
| `--warped_noise_probability` | Fraction of steps using warped noise (default `1.0`) |

Training was run on **8× A100 80GB GPUs** using DeepSpeed ZeRO stage 2.

</details>

---

## 🤩 Community Adoption

We are excited to see the community build on VOID!  
Below we showcase selected demos, tools, and extensions.

If you’ve built something using VOID, feel free to submit a PR to add it here.

### 🌐 Demos & Projects

- ⭐ **Gradio Demo** — @sam-motamed  
  Interactive demo for trying VOID in the browser:  
  👉 https://huggingface.co/spaces/sam-motamed/VOID

## 🙏 Acknowledgements

This implementation builds on code and models from [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main), [Gen-Omnimatte](https://github.com/gen-omnimatte/gen-omnimatte-public/tree/main), [Go-with-the-Flow](https://github.com/Eyeline-Labs/Go-with-the-Flow), [Kubric](https://github.com/google-research/kubric) and [HUMOTO](https://jiaxin-lu.github.io/humoto/). We thank the authors for sharing the codes and pretrained inpainting models for CogVideoX, Gen-Omnimatte, and the optical flow warping utilities.

---
## Star History

<a href="https://www.star-history.com/?repos=netflix%2Fvoid-model&type=date&legend=bottom-right">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=netflix/void-model&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=netflix/void-model&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=netflix/void-model&type=date&legend=top-left" />
 </picture>
</a>

---
## 📄 Citation

If you find our work useful, please consider citing:

🔗 https://arxiv.org/abs/2604.02296

```bibtex
@misc{motamed2026void,
  title={VOID: Video Object and Interaction Deletion},
  author={Saman Motamed and William Harvey and Benjamin Klein and Luc Van Gool and Zhuoning Yuan and Ta-Ying Cheng},
  year={2026},
  eprint={2604.02296},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.02296}
}
