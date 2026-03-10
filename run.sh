#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

INPUT_DIR=${1:-data/submission_images}
OUTPUT_DIR=${2:-retinex2_output}

mkdir -p "$OUTPUT_DIR"
echo "INPUT_DIR  = $INPUT_DIR"
echo "OUTPUT_DIR = $OUTPUT_DIR"

for img in "$INPUT_DIR"/*.tif; do
    echo "Processing $img"

    python "$REPO_ROOT/src/retinex2_corrected.py" \
        --beta 100 \
        --max_abs_alpha 0.9 \
        --gamma 6.2 \
        --input "$img" \
        --output_dir "$OUTPUT_DIR" \
        --use_model \
        --chatgpt
done
