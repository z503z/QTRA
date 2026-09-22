#!/usr/bin/env bash
set -euo pipefail
: "${CDVQA_ROOT:?Set CDVQA_ROOT to the question/answer JSON directory}"
: "${SECOND_ROOT:?Set SECOND_ROOT to the SECOND image root}"
: "${CLIP_WEIGHTS:?Set CLIP_WEIGHTS to RN101.pt}"
: "${X3D_WEIGHTS:?Set X3D_WEIGHTS to X3D_L.pyth}"
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/cdvqa}
"${PYTHON_BIN}" train.py --config configs/cdvqa.json \
  --cdvqa-root "${CDVQA_ROOT}" --second-root "${SECOND_ROOT}" \
  --clip-pretrained "${CLIP_WEIGHTS}" --x3d-pretrained "${X3D_WEIGHTS}" \
  --seed "${SEED:-42}" --output "${OUTPUT_ROOT}"
