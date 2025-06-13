#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <audio_filename> <output_name> <feature_str>"
    exit 1
fi

AUDIO_PATH="$1"
NAME="$2"
FEATURE_STR="$3"

# Run the Python script
python -m snap2midi.train_scripts.shallow_transcriber.shallow_inference \
    --config_path "./confs/shallow_inference_config.json" \
    --name "$NAME" \
    --audio_path "$AUDIO_PATH" \
    --feature_str "$FEATURE_STR"

