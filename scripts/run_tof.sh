#!/usr/bin/env bash
# ToF ECG 시각화 파이프라인: 임베딩 + UMAP + 연령대별 UMAP
# 사용법: bash scripts/run_tof.sh [GPU_IDS] [BATCH_SIZE]
#   예)   bash scripts/run_tof.sh 0     128
#        bash scripts/run_tof.sh 0,1   256

set -euo pipefail

GPUS="${1:-0}"
BATCH="${2:-128}"
CONFIG="configs/tof_ecg.json"
TAG="tof_$(date +%Y%m%d_%H%M%S)"

cd "$(dirname "$0")/.."

echo "[1/2] 임베딩 + 데이터셋/라벨 UMAP → results/${TAG}"
python scripts/run_all_embedding_umap.py \
    --config "${CONFIG}" \
    --n_samples 0 \
    --batch_size "${BATCH}" \
    --gpus "${GPUS}" \
    --tag "${TAG}"

echo "[2/2] 연령대별 UMAP (balanced)"
python scripts/plot_age_umap.py \
    --run_dir "results/${TAG}" \
    --balanced

echo
echo "완료: results/${TAG}"
ls "results/${TAG}"
