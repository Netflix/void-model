#!/bin/bash
# run_pipeline.sh
# Runs stages 1-4 given a points config JSON (output of point_selector_gui.py)
#
# Usage:
#   bash run_pipeline.sh <config_points.json> [OPTIONS]
#
# Options:
#   --sam2-checkpoint PATH          SAM2 checkpoint for stage 1 (default: ../sam2_hiera_large.pt)
#   --device DEVICE                 cuda or cpu (default: cuda)
#   --stage3-segmentation-model MODEL  sam3 or langsam for stage 3a (default: sam3)
#
# Examples:
#   bash run_pipeline.sh my_config_points.json
#   bash run_pipeline.sh my_config_points.json --sam2-checkpoint ../sam2_hiera_large.pt
#   bash run_pipeline.sh my_config_points.json --stage3-segmentation-model langsam

set -e

# ── Arguments ──────────────────────────────────────────────────────────────────
CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "Usage: bash run_pipeline.sh <config_points.json> [OPTIONS]"
    echo "See script header for available options."
    exit 1
fi

SAM2_CHECKPOINT="../sam2_hiera_large.pt"
DEVICE="cuda"
STAGE3_SEGMENTATION_MODEL="sam3"

# Parse optional flags
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sam2-checkpoint)          SAM2_CHECKPOINT="$2"; shift 2 ;;
        --device)                   DEVICE="$2";          shift 2 ;;
        --stage3-segmentation-model) STAGE3_SEGMENTATION_MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Void Mask Generation Pipeline"
echo "=========================================="
echo "Config:                   $CONFIG"
echo "SAM2 checkpoint:          $SAM2_CHECKPOINT"
echo "Device:                   $DEVICE"
echo "Stage 3 segmentation:     $STAGE3_SEGMENTATION_MODEL"
echo "=========================================="

# ── Stage 1: SAM2 Segmentation ─────────────────────────────────────────────────
echo ""
echo "[1/4] SAM2 segmentation..."
python "$SCRIPT_DIR/stage1_sam2_segmentation.py" \
    --config "$CONFIG" \
    --sam2-checkpoint "$SAM2_CHECKPOINT" \
    --device "$DEVICE"

# ── Stage 2: VLM Analysis ──────────────────────────────────────────────────────
echo ""
echo "[2/4] VLM analysis (Gemini)..."
python "$SCRIPT_DIR/stage2_vlm_analysis.py" \
    --config "$CONFIG"

# ── Stage 3a: Generate Grey Masks ─────────────────────────────────────────────
echo ""
echo "[3/4] Generating grey masks..."
python "$SCRIPT_DIR/stage3a_generate_grey_masks_v2.py" \
    --config "$CONFIG" \
    --segmentation-model "$STAGE3_SEGMENTATION_MODEL"

# ── Stage 4: Combine into Quadmask ────────────────────────────────────────────
echo ""
echo "[4/4] Combining masks into quadmask_0.mp4..."
python "$SCRIPT_DIR/stage4_combine_masks.py" \
    --config "$CONFIG"

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Output: quadmask_0.mp4 in each video's output_dir"
echo "=========================================="
