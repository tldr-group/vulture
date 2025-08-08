#!/bin/bash

# download checkpoints from huggingface with curl
# 1) get json of repo 2) filter to .pth 3) download 
USERNAME="rmdocherty"
MODEL_ID="vulture"
TARGET_DIR="trained_models"
# this is a token with read-only access to the repo
HF_TOKEN="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"

mkdir -p "$TARGET_DIR"

# Get list of files from Hugging Face API
curl -s -H "Authorization: Bearer $HF_TOKEN" \
 "https://huggingface.co/api/models/$USERNAME/$MODEL_ID" \
    | grep -oP '"rfilename"\s*:\s*"\K[^"]+' \
    | grep '\.pth' \
    | while read FILE; do
        URL="https://huggingface.co/$USERNAME/$MODEL_ID/resolve/main/$FILE"
        OUT_FILE="$TARGET_DIR/$FILE"
        echo "Downloading $FILE ..."
        curl -s -L -H "Authorization: Bearer $HF_TOKEN" "$URL" -o "$OUT_FILE"
    done