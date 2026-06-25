#!/usr/bin/env bash
# 4-패널 통합 UMAP 그리드
#   col 0: Adult (PTB-XL) vs Pediatric (ZZU)
#   col 1: Normal vs Abnormal (combined)
#   col 2: Adult Normal vs Abnormal (PTB-XL only)
#   col 3: Pediatric Normal vs Abnormal (ZZU only)
# 모든 패널은 1:1 다운샘플링 후 UMAP fit.
#
# 입력: visuallize/results/embeddings/  (기본)
# 출력: visuallize/results/<timestamp>/umap_4panel.png
#
# 사용법:
#   bash scripts/run_age_split.sh
#   bash scripts/run_age_split.sh <EMBEDDINGS_DIR>

set -euo pipefail

cd "$(dirname "$0")/.."

EMB_DIR="${1:-$(pwd)/results/embeddings}"

if [[ ! -d "${EMB_DIR}" ]]; then
    echo "ERROR: embeddings 디렉토리가 없습니다: ${EMB_DIR}" >&2
    exit 1
fi

# 결과는 results/<timestamp>/ 아래에 (embeddings + UMAP coord 캐시는 공유)
RESULTS_ROOT="$(dirname "${EMB_DIR}")"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${RESULTS_ROOT}/${TS}"
mkdir -p "${OUT_DIR}"

echo "================================================================"
echo "  embeddings_dir : ${EMB_DIR}"
echo "  output_dir     : ${OUT_DIR}"
echo "================================================================"

python scripts/plot_umap_4panel.py \
    --embeddings_dir "${EMB_DIR}" \
    --output_dir "${OUT_DIR}"

echo
echo "완료. 생성 파일:"
ls -lh "${OUT_DIR}"/umap_4panel.png 2>/dev/null || true
