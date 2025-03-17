#!/bin/bash

# Directory to scan (default to current directory if not provided as argument)
DIR=${1:-.}
DIR=$(realpath "$DIR")  # Convert to absolute path

# List of keywords to check
KEYWORDS=(
    "stack_d"
    "stack_three_d"
    "coffee_d"
    "threading_d"
    "three_piece_assembly_d"
    "hammer_cleanup_d"
    "mug_cleanup_d"
    "kitchen_d"
    "coffee_preparation_d"
)

# Iterate over files in the directory
for file in "$DIR"/*; do
    filename=$(basename "$file")
    for keyword in "${KEYWORDS[@]}"; do
        if [[ "$filename" == *"$keyword"* ]]; then
            bn=$(basename "$file")
            bn_no_ext="${file%.*}"
            output_file="${bn_no_ext}_im170.hdf5"
            if [[ -f "$output_file" ]]; then
                echo "Skipping $file as $output_file already exists."
                continue
            fi
            echo "Processing file: $file"
            OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            python robomimic/scripts/dataset_states_to_obs.py \
                --dataset "$file" \
                --n 1000 \
                --camera_width 170 \
                --camera_height 128 \
                --num_procs 20 \
                --camera_names robot0_agentview robot0_eye_in_hand \
                --postprocess_model_xml
            # python robomimic/scripts/playback_dataset.py --dataset "${output_file}" --use-obs --video_path "${bn_no_ext}.mp4" --n 5 --render_image_names robot0_agentview robot0_eye_in_hand
        fi
    done
done

wait