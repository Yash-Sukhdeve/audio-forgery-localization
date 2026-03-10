#!/bin/bash
# Train BAM on PartialSpoof
set -e

PROJECT_DIR="/media/lab2208/ssd/Explainablility/Localization"
BAM_DIR="${PROJECT_DIR}/baselines/repos/BAM"
DATA_DIR="${BAM_DIR}/data"
WAVLM_CKPT="${PROJECT_DIR}/checkpoints/wavlm_large.pt"

echo "=== BAM Training ==="
echo "BAM repo: ${BAM_DIR}"
echo "Data: ${DATA_DIR}"

# Prepare data if not done
if [ ! -L "${DATA_DIR}/raw/train" ]; then
    echo "Preparing data..."
    python "${PROJECT_DIR}/baselines/wrappers/bam_data_prep.py" \
        --ps_root /media/lab2208/ssd/datasets/PartialSpoof/database \
        --output_dir "${DATA_DIR}" \
        --resolution 0.02
fi

# Train
cd "${BAM_DIR}"
python train.py \
    --exp_name bam_wavlm_ps \
    --train_root "${DATA_DIR}/raw/train" \
    --dev_root "${DATA_DIR}/raw/dev" \
    --eval_root "${DATA_DIR}/raw/eval" \
    --label_root "${DATA_DIR}" \
    --max_epochs 50 \
    --batch_size 8 \
    --base_lr 1e-5 \
    --weight_decay 1e-4 \
    --samplerate 16000 \
    --resolution 0.02 \
    --gpu "[0]"

echo "=== BAM Training Complete ==="
