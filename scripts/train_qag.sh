#!/usr/bin/env bash
# Fresh validation-only training pipeline. Run from the repository root.
set -euo pipefail
: "${QAG_ROOT:?Set QAG_ROOT to the directory containing train and val}"
: "${CLIP_WEIGHTS:?Set CLIP_WEIGHTS to RN101.pt}"
: "${X3D_WEIGHTS:?Set X3D_WEIGHTS to X3D_L.pyth}"
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/qag}
SEED=${SEED:-42}
COMMON=(--train-root "${QAG_ROOT}/train" --val-root "${QAG_ROOT}/val"
        --clip-pretrained "${CLIP_WEIGHTS}" --seed "${SEED}")

"${PYTHON_BIN}" train.py --config configs/qag_baseline.json "${COMMON[@]}" \
  --x3d-pretrained "${X3D_WEIGHTS}" --output "${OUTPUT_ROOT}/baseline"
"${PYTHON_BIN}" train.py --config configs/qag_qtra.json "${COMMON[@]}" \
  --init-checkpoint "${OUTPUT_ROOT}/baseline/best_val_oa.pth" --output "${OUTPUT_ROOT}/qtra"
"${PYTHON_BIN}" train.py --config configs/qag_global.json "${COMMON[@]}" \
  --init-checkpoint "${OUTPUT_ROOT}/qtra/best_val_miou.pth" --output "${OUTPUT_ROOT}/global"
"${PYTHON_BIN}" train.py --config configs/qag_grounding.json "${COMMON[@]}" \
  --init-checkpoint "${OUTPUT_ROOT}/global/best_val_answer.pth" --output "${OUTPUT_ROOT}/grounding"
"${PYTHON_BIN}" train.py --config configs/qag_full.json "${COMMON[@]}" \
  --init-checkpoint "${OUTPUT_ROOT}/grounding/best_val_grounding.pth" --output "${OUTPUT_ROOT}/full"
