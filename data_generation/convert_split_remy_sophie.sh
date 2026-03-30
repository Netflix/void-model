#!/bin/bash
# Convert 699 scenarios: Half with Remy, half with Sophie
# More efficient: 699 conversions instead of 1398

# Don't use set -e to allow conversion to continue even with minor errors
#set -e

# Get absolute path of project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths (now absolute)
HUMOTO_DIR="$PROJECT_ROOT/humoto_release/humoto_0805"
OUTPUT_BASE="$PROJECT_ROOT/humoto_characters_converted"
REMY_FBX="$PROJECT_ROOT/human_model/Remy_mixamo_bone.fbx"
SOPHIE_FBX="$PROJECT_ROOT/human_model/Sophie_mixamo_bone.fbx"
LOG_FILE="$PROJECT_ROOT/conversion_split.log"
PROGRESS_FILE="$PROJECT_ROOT/conversion_split_progress.txt"

# Get all scenarios
scenarios=($(ls -1 "$HUMOTO_DIR" | sort))
total=${#scenarios[@]}
half=$((total / 2))

echo "============================================================" | tee -a "$LOG_FILE"
echo "Converting $total scenarios: Split between Remy & Sophie" | tee -a "$LOG_FILE"
echo "Remy: First $half scenarios" | tee -a "$LOG_FILE"
echo "Sophie: Last $((total - half)) scenarios" | tee -a "$LOG_FILE"
echo "Total conversions: $total (instead of $((total * 2)))" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Load progress if exists
declare -A completed
if [ -f "$PROGRESS_FILE" ]; then
    while IFS='|' read -r char seq status; do
        completed["${char}|${seq}"]="$status"
    done < "$PROGRESS_FILE"
    echo "Loaded $(wc -l < "$PROGRESS_FILE") completed conversions" | tee -a "$LOG_FILE"
fi

completed_count=0
failed_count=0
skipped_count=0

# Process all scenarios
for idx in "${!scenarios[@]}"; do
    seq="${scenarios[$idx]}"

    # Determine character: first half = Remy, second half = Sophie
    if [ $idx -lt $half ]; then
        char_name="remy"
        char_fbx="$REMY_FBX"
    else
        char_name="sophie"
        char_fbx="$SOPHIE_FBX"
    fi

    # Check if already completed
    if [ "${completed[${char_name}|${seq}]}" == "completed" ]; then
        echo "[$((idx+1))/$total] $seq ($char_name): SKIPPED" | tee -a "$LOG_FILE"
        ((skipped_count++))
        continue
    fi

    echo "[$((idx+1))/$total] Converting $seq with $char_name..." | tee -a "$LOG_FILE"

    original_path="$HUMOTO_DIR/$seq"
    char_output="$OUTPUT_BASE/$char_name"

    # Step directories (scripts create nested subdirectories with scenario name)
    step1_base="${OUTPUT_BASE}_step1/${char_name}/${seq}"
    step2_base="${OUTPUT_BASE}_step2/${char_name}/${seq}"
    step3_base="${OUTPUT_BASE}_step3/${char_name}/${seq}"
    step1_dir="$step1_base/$seq"
    step2_dir="$step2_base/$seq"
    step3_dir="$step3_base/$seq"
    final_dir="$char_output/$seq"

    mkdir -p "$step1_base" "$step2_base" "$step3_base" "$final_dir"

    # Step 1: Clear scale
    echo "  [1/4] Clearing scale..." | tee -a "$LOG_FILE"
    cd "$PROJECT_ROOT/scripts"
    if python clear_human_scale.py -d "$original_path" -o "$step1_base" >> "$LOG_FILE" 2>&1; then
        echo "  ✓ Step 1" | tee -a "$LOG_FILE"
    else
        echo "  ✗ Step 1 FAILED" | tee -a "$LOG_FILE"
        cd "$PROJECT_ROOT"
        echo "${char_name}|${seq}|failed" >> "$PROGRESS_FILE"
        ((failed_count++))
        continue
    fi
    cd "$PROJECT_ROOT"

    # Step 2: Transfer character
    echo "  [2/4] Transferring $char_name..." | tee -a "$LOG_FILE"
    cd "$PROJECT_ROOT/scripts"
    if python transfer_human_model.py -d "$step1_dir" -o "$step2_base" --human_model "$char_fbx" >> "$LOG_FILE" 2>&1; then
        echo "  ✓ Step 2" | tee -a "$LOG_FILE"
    else
        echo "  ✗ Step 2 FAILED" | tee -a "$LOG_FILE"
        cd "$PROJECT_ROOT"
        echo "${char_name}|${seq}|failed" >> "$PROGRESS_FILE"
        ((failed_count++))
        continue
    fi
    cd "$PROJECT_ROOT"

    # Step 3: Extract pickle
    echo "  [3/4] Extracting pickle..." | tee -a "$LOG_FILE"
    cd "$PROJECT_ROOT/scripts"
    if python extract_pk_data.py -d "$step2_dir" -o "$step3_base" >> "$LOG_FILE" 2>&1; then
        echo "  ✓ Step 3" | tee -a "$LOG_FILE"
    else
        echo "  ✗ Step 3 FAILED" | tee -a "$LOG_FILE"
        cd "$PROJECT_ROOT"
        echo "${char_name}|${seq}|failed" >> "$PROGRESS_FILE"
        ((failed_count++))
        continue
    fi
    cd "$PROJECT_ROOT"

    # Step 4: Copy metadata and final files
    echo "  [4/4] Copying metadata and files..." | tee -a "$LOG_FILE"

    # Copy YAML file from original
    if [ -f "$original_path/$seq.yaml" ]; then
        cp "$original_path/$seq.yaml" "$final_dir/" >> "$LOG_FILE" 2>&1
    fi

    # Copy FBX from step2 (character with animation)
    # Note: Scripts create nested directories, so files are at step2_dir/seq/seq.fbx
    if [ -f "$step2_dir/$seq/$seq.fbx" ]; then
        cp "$step2_dir/$seq/$seq.fbx" "$final_dir/" >> "$LOG_FILE" 2>&1
    elif [ -f "$step2_dir/$seq.fbx" ]; then
        cp "$step2_dir/$seq.fbx" "$final_dir/" >> "$LOG_FILE" 2>&1
    fi

    # Copy PKL from step3 (extracted skeleton data)
    # Note: Scripts create nested directories, so files are at step3_dir/seq/seq.pkl
    if [ -f "$step3_dir/$seq/$seq.pkl" ]; then
        cp "$step3_dir/$seq/$seq.pkl" "$final_dir/" >> "$LOG_FILE" 2>&1
        echo "  ✓ Step 4" | tee -a "$LOG_FILE"
    elif [ -f "$step3_dir/$seq.pkl" ]; then
        cp "$step3_dir/$seq.pkl" "$final_dir/" >> "$LOG_FILE" 2>&1
        echo "  ✓ Step 4" | tee -a "$LOG_FILE"
    else
        echo "  ✗ Step 4 FAILED: step3 output not found" | tee -a "$LOG_FILE"
        echo "${char_name}|${seq}|failed" >> "$PROGRESS_FILE"
        ((failed_count++))
        continue
    fi

    echo "  ✓ $seq completed with $char_name!" | tee -a "$LOG_FILE"
    echo "${char_name}|${seq}|completed" >> "$PROGRESS_FILE"
    ((completed_count++))

    # Progress every 10
    if [ $(((idx+1) % 10)) -eq 0 ]; then
        echo "  Progress: $completed_count completed, $failed_count failed, $skipped_count skipped" | tee -a "$LOG_FILE"
    fi
done

# Final summary
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Conversion Complete!" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Completed: $completed_count" | tee -a "$LOG_FILE"
echo "Failed: $failed_count" | tee -a "$LOG_FILE"
echo "Skipped: $skipped_count" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Remy scenarios: $half" | tee -a "$LOG_FILE"
echo "Sophie scenarios: $((total - half))" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_BASE/" | tee -a "$LOG_FILE"
echo "Progress: $PROGRESS_FILE" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
