#!/bin/bash

# Minimum required free GPU memory in GB
min_memory=20000

# Read commands from text file
IFS=$'\n' commands=($(< pending_retraining.txt))

# GPU ids to be considered
GPU_IDS="2 3 4 7"
gpu_ids=${GPU_IDS:-"0 1 2 3"}

for cmd in "${commands[@]}"; do
    while true; do
        echo $cmd
        # Get the GPU with at least min_memory free memory and GPU index more than 1
        gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv | awk -F "," -v gpu_ids="$gpu_ids" '{ if (match(gpu_ids,$1)) print $0 }' | awk -F "," -v min_memory=15000 '$2+0 > min_memory { print $1 }' | shuf -n 1)
        
        if [ -z $gpu ]; then
            # If no GPU is available then sleep for 5 seconds and check again
            echo "All GPUs are full, waiting for a GPU to become available..."
            sleep 5
        else
            break
        fi
    done
    # Launch command on the selected GPU
    eval CUDA_VISIBLE_DEVICES=$gpu $cmd &
    sleep 30
done

wait
