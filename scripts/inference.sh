export PYTHONPATH=${PYTHONPATH}:./main/InternVL-2

export MODEL_PATH=${MODEL_PATH:-path_to/your_mm_rm/checkpoint-last}

export DATA_PATH="path_to/sample_data.jsonl"
export DEST_FILE="path_to/sample_data_score.jsonl"
export PER_GPUS=${PER_GPUS:-1}

# keep same as the training setting
export CONV=${CONV:-internlm2-chat}
export PATCHS=${PATCHS:-6}
export DYNAMIC_PATCH=${DYNAMIC_PATCH:-true}
export USE_THUMBNAIL=${USE_THUMBNAIL:-true}
export MAX_LENGTH=${MAX_LENGTH:-4096}

python main/InternVL-2/sample_score.py  \
    --model_name_or_path "$MODEL_PATH" \
    --max_dynamic_patch $PATCHS \
    --conv_style "$CONV" \
    --dynamic_image_size $DYNAMIC_PATCH \
    --use_thumbnail $USE_THUMBNAIL \
    --max_length $MAX_LENGTH \
    --bf16 true --tf32 true --output_dir None
