# VLM Mask Reasoner — Mask Generation Pipeline

Generates quadmasks for video inpainting by combining user-guided SAM2 segmentation with VLM (Gemini) scene reasoning. The output `quadmask_0.mp4` encodes four semantic layers that the inpainting model uses to understand what to remove and what to preserve.

---

## Quadmask Values

| Value | Meaning |
|-------|---------|
| `0` | Primary object (to be removed) |
| `63` | Overlap of primary and affected regions |
| `127` | Affected objects (shadows, reflections, held items) |
| `255` | Background — keep as-is |

---

## Step 1 — Select Points via GUI

Launch the point selector GUI:

```bash
python point_selector_gui.py
```

Use this GUI to place sparse click points on the object(s) you want removed.

**A few things to know:**

- Points can be placed on **any frame**, not just the first. If the object you want to remove only appears later in the video, navigate to that frame and click there.
- You can place points across **multiple frames** — useful when there are multiple distinct objects to remove, or when an object's position shifts significantly over time.
- The GUI saves your selections to a `config_points.json` file. Keep track of where this is saved — you'll pass it to the pipeline next.

---

## Step 2 — Run the Pipeline

Once you have your `config_points.json`, run all stages with a single command:

```bash
bash run_pipeline.sh <config_points.json>
```

Optional flags:

```bash
bash run_pipeline.sh <config_points.json> \
    --sam2-checkpoint ../sam2_hiera_large.pt \    --stage3-segmentation-model langsam \    --device cuda
```

This runs four stages automatically:

1. **Stage 1 — SAM2 Segmentation:** Propagates your point clicks into a per-frame black mask for the primary object.
2. **Stage 2 — VLM Analysis (Gemini):** Analyzes the scene to identify affected objects — things like shadows, reflections, or items the primary object is interacting with.
3. **Stage 3 — Grey Mask Generation:** Produces a grey mask track for the affected objects identified in Stage 2.
4. **Stage 4 — Combine into Quadmask:** Merges the black and grey masks into the final `quadmask_0.mp4`.

The output `quadmask_0.mp4` is written into each video's `output_dir` as specified in the config.

> **Note on grey values in frame 1:** The inpainting model was trained with grey-valued regions (`127`) starting from frame 1 onward — not on the very first frame. We find this convention improves inference quality, so the pipeline automatically clears any grey pixels from frame 0 of the final quadmask before saving.

---

## Step 3 (Optional) — Manual Mask Correction

If the generated quadmask needs refinement, you can correct it interactively:

```bash
python edit_quadmask.py
```

Point the GUI to the folder containing `quadmask_0.mp4`. You can paint over regions frame-by-frame to fix any mask errors before running inference. The corrected mask is saved back to `quadmask_0.mp4` in the same folder.

---

## Installation & Dependencies

### 1. Python dependencies

Install the main requirements from the repo root:

```bash
pip install -r requirements.txt
```

### 2. SAM2 (Stage 1)

SAM2 must be installed separately (it is not on PyPI):

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

Then download the SAM2 checkpoint. The pipeline defaults to `sam2_hiera_large.pt` one level above this directory:

```bash
# from the repo root (or wherever you want to store it)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### 3. SAM3 or LangSAM (Stage 3a)

Stage 3a uses text-prompted segmentation to identify affected objects. You need **either** SAM3 or LangSAM:

**Option A: SAM3 (default, recommended)**

SAM3 requires Python 3.12+, PyTorch 2.7+, and HuggingFace authentication:

```bash
# Install SAM3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..
```

**Checkpoint access:** SAM3 checkpoints are hosted on HuggingFace and require authentication:

1. Request access at https://huggingface.co/facebook/sam3.1
2. Once approved, authenticate:
   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login  # Enter your HF token
   ```
3. The model will auto-download checkpoints on first use

See the [SAM3 repo](https://github.com/facebookresearch/sam3) for full installation details.

**Option B: LangSAM (alternative)**

LangSAM combines SAM 2.1 with GroundingDINO for text-prompted segmentation. Requires Python 3.10+:

```bash
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```

LangSAM auto-downloads its checkpoints (GroundingDINO + SAM 2.1) and doesn't require authentication. To use it, pass `--stage3-segmentation-model langsam` to `run_pipeline.sh`.

If you place the checkpoint elsewhere, pass it explicitly:

```bash
bash run_pipeline.sh config_points.json --sam2-checkpoint /path/to/sam2_hiera_large.pt
```

> SAM2 requires **Python ≥ 3.10** and **PyTorch ≥ 2.3.1** with CUDA. See the [SAM2 repo](https://github.com/facebookresearch/segment-anything-2) for full system requirements.

### 3. Gemini API key

Stage 2 uses the Gemini VLM. Export your API key before running the pipeline:

```bash
export GEMINI_API_KEY="your_key_here"
```
