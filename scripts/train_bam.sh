#!/bin/bash
# Train BAM on PartialSpoof
# BAM native resolution: 0.16s (160ms), matching their ps_preprocess.py
set -e

PROJECT_DIR="/media/lab2208/ssd/Explainablility/Localization"
BAM_DIR="${PROJECT_DIR}/baselines/repos/BAM"
DATA_DIR="${BAM_DIR}/data"
WAVLM_CKPT="${PROJECT_DIR}/checkpoints/wavlm_large.pt"

# Parse optional args (--max_epochs, --batch_size, etc.)
MAX_EPOCHS="${1:-50}"

echo "=== BAM Training ==="
echo "BAM repo: ${BAM_DIR}"
echo "Data: ${DATA_DIR}"
echo "Max epochs: ${MAX_EPOCHS}"

# Prepare data if not done
if [ ! -d "${DATA_DIR}/boundary_0.16_labels" ]; then
    echo "Preparing data..."
    python "${PROJECT_DIR}/baselines/wrappers/bam_data_prep.py" \
        --ps_root /media/lab2208/ssd/datasets/PartialSpoof/database \
        --output_dir "${DATA_DIR}" \
        --resolution 0.16
fi

# Copy config override (runtime artifact, not modifying repo source)
cp "${PROJECT_DIR}/configs/bam_wavlm_ps.yaml" "${BAM_DIR}/config/bam_wavlm_ps.yaml"

# Train
# NOTE: --base_lr and --weight_decay cannot be passed via CLI due to
# argparse type=int bug in BAM's train.py. Defaults (1e-5, 1e-4) match
# the paper specification, so we rely on those.
# NOTE: --resolution cannot be passed via CLI for same reason (type=int).
# Default is 0.16 which is BAM's native resolution.
cd "${BAM_DIR}"
python train.py \
    --exp_name bam_wavlm_ps \
    --train_root "${DATA_DIR}/raw/train" \
    --dev_root "${DATA_DIR}/raw/dev" \
    --eval_root "${DATA_DIR}/raw/eval" \
    --label_root "${DATA_DIR}" \
    --max_epochs "${MAX_EPOCHS}" \
    --batch_size 8 \
    --samplerate 16000

echo "=== BAM Training Complete ==="
