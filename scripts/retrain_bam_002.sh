#!/bin/bash
# Retrain BAM at 0.02s (20ms) resolution for fair comparison with FARA.
#
# BAM's native resolution is 0.16s (160ms). FARA evaluates all methods at
# 0.02s. This script retrains BAM at 0.02s for apples-to-apples comparison.
#
# Key differences from the 0.16s training (train_bam.sh):
#   1. Resolution 0.02 instead of 0.16 (8x finer)
#   2. label_maxlength scaled 25 -> 200 (same ~4s max audio)
#   3. Workaround for BAM's argparse type=int bug on --resolution
#      (passing 0.02 via CLI would error; we use a Python wrapper)
#   4. Separate config YAML: bam_wavlm_ps_002.yaml
#   5. Separate experiment name: bam_wavlm_ps_002
#
# References:
#   - Zhang et al., "Partially Fake Audio Detection," ICASSP 2021
#     (PartialSpoof segment labels at multiple resolutions)
#   - BAM: https://github.com/Yongbink/BAM (boundary-aware model)
#
# Usage:
#   bash scripts/retrain_bam_002.sh [MAX_EPOCHS]
#
# NOTE: Do NOT modify the BAM repo. All workarounds are external.
set -euo pipefail

###############################################################################
# Paths
###############################################################################
PROJECT_DIR="/media/lab2208/ssd/Explainablility/Localization"
BAM_DIR="${PROJECT_DIR}/baselines/repos/BAM"
DATA_DIR="${BAM_DIR}/data"
WAVLM_CKPT="${PROJECT_DIR}/checkpoints/wavlm_large.pt"
PS_ROOT="/media/lab2208/ssd/datasets/PartialSpoof/database"
AFL_DIR="/media/lab2208/ssd/audio-forgery-localization"

RESOLUTION="0.02"
EXP_NAME="bam_wavlm_ps_002"
MAX_EPOCHS="${1:-50}"
SAMPLERATE=16000
# At 0.02s, sequences are 8x longer than 0.16s (200 vs 25 frames).
# BAM's graph attention is O(n^2), so reduce batch size to avoid OOM.
# RTX 4080 16GB: batch_size=2 fits with 0.02s resolution.
BATCH_SIZE=2
GRAD_ACCUM=4  # Effective batch size: 2 * 4 = 8 (same as original)
# At 0.16s, label_maxlength=25 covers 25*0.16=4s of audio.
# At 0.02s, same 4s requires 4/0.02=200 label frames.
LABEL_MAXLENGTH=200

echo "=== BAM Retrain at ${RESOLUTION}s Resolution ==="
echo "BAM repo:        ${BAM_DIR}"
echo "Data dir:        ${DATA_DIR}"
echo "PartialSpoof:    ${PS_ROOT}"
echo "Resolution:      ${RESOLUTION}s"
echo "Label maxlength: ${LABEL_MAXLENGTH}"
echo "Max epochs:      ${MAX_EPOCHS}"
echo "Experiment:      ${EXP_NAME}"

###############################################################################
# Step 1: Verify 0.02 segment labels exist in PartialSpoof
###############################################################################
echo ""
echo "--- Step 1: Verifying PartialSpoof 0.02 labels ---"
for split in train dev eval; do
    label_file="${PS_ROOT}/segment_labels/${split}_seglab_${RESOLUTION}.npy"
    if [ ! -f "${label_file}" ]; then
        echo "ERROR: Missing ${label_file}"
        exit 1
    fi
    echo "  Found: ${label_file}"
done

###############################################################################
# Step 2: Prepare BAM data at 0.02s resolution
###############################################################################
echo ""
echo "--- Step 2: Preparing data at ${RESOLUTION}s resolution ---"
if [ ! -d "${DATA_DIR}/boundary_${RESOLUTION}_labels" ]; then
    python "${AFL_DIR}/baselines/wrappers/bam_data_prep.py" \
        --ps_root "${PS_ROOT}" \
        --output_dir "${DATA_DIR}" \
        --resolution "${RESOLUTION}"
else
    echo "  Boundary labels already exist at ${DATA_DIR}/boundary_${RESOLUTION}_labels, skipping."
fi

###############################################################################
# Step 3: Create config YAML for the 0.02 experiment
#         Uses same model hyperparams as the 0.16 config, only name differs.
#         We do NOT modify the BAM repo; config is placed where BAM expects it.
###############################################################################
echo ""
echo "--- Step 3: Creating config ${EXP_NAME}.yaml ---"
cat > "${BAM_DIR}/config/${EXP_NAME}.yaml" <<YAML
embed_dim: 1024
gap_head_num: 1
gap_layer_num: 2
local_channel_dim: 32
pool_head_num: 1
ssl_ckpt: ${WAVLM_CKPT}
ssl_feat_dim: 1024
ssl_name: wavlm_local
YAML
echo "  Written: ${BAM_DIR}/config/${EXP_NAME}.yaml"

###############################################################################
# Step 4: Train BAM at 0.02s resolution
#
# WORKAROUND for BAM's argparse type=int bug on --resolution:
#   BAM's train.py declares --resolution with type=int, so passing 0.02
#   via the command line would call int("0.02") which raises ValueError.
#   Similarly, --base_lr and --weight_decay are type=int.
#
#   Solution: monkey-patch argparse.ArgumentParser.parse_args to fix the
#   float values after parsing, then execute train.py via runpy.
#   This avoids any modification to the BAM repository.
###############################################################################
echo ""
echo "--- Step 4: Training BAM (${EXP_NAME}) ---"
cd "${BAM_DIR}"

python -c "
import argparse
import sys

# Monkey-patch parse_args to fix the type=int bug for float arguments
_original_parse_args = argparse.ArgumentParser.parse_args

def _patched_parse_args(self, args=None, namespace=None):
    result = _original_parse_args(self, args, namespace)
    # Fix resolution: type=int truncates 0.02 -> 0, but we need float
    if hasattr(result, 'resolution'):
        result.resolution = ${RESOLUTION}
    # Fix base_lr and weight_decay (also type=int in BAM)
    if hasattr(result, 'base_lr'):
        result.base_lr = 1e-5
    if hasattr(result, 'weight_decay'):
        result.weight_decay = 1e-4
    return result

argparse.ArgumentParser.parse_args = _patched_parse_args

# Monkey-patch Lightning Trainer to add gradient accumulation
# This compensates for reduced batch_size (2 * 4 = effective 8)
import lightning as L
_orig_trainer_init = L.Trainer.__init__
def _patched_trainer_init(self, *args, **kwargs):
    kwargs['accumulate_grad_batches'] = ${GRAD_ACCUM}
    return _orig_trainer_init(self, *args, **kwargs)
L.Trainer.__init__ = _patched_trainer_init

# Set sys.argv as if we called train.py directly
sys.argv = [
    'train.py',
    '--exp_name', '${EXP_NAME}',
    '--train_root', '${DATA_DIR}/raw/train',
    '--dev_root', '${DATA_DIR}/raw/dev',
    '--eval_root', '${DATA_DIR}/raw/eval',
    '--label_root', '${DATA_DIR}',
    '--max_epochs', '${MAX_EPOCHS}',
    '--batch_size', '${BATCH_SIZE}',
    '--samplerate', '${SAMPLERATE}',
    '--label_maxlength', '${LABEL_MAXLENGTH}',
]

# Execute train.py via runpy (equivalent to 'python train.py ...')
import runpy
runpy.run_path('train.py', run_name='__main__')
"

echo "=== BAM Retrain at ${RESOLUTION}s Complete ==="
