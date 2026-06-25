#!/usr/bin/env bash
# 기존 임베딩으로 라벨별 / Normal-vs-Any-Abnormal UMAP 재생성
# 사용법:
#   bash scripts/run_replot.sh <RUN_DIR>
#     예) bash scripts/run_replot.sh results/tof_20260423_094430
#   bash scripts/run_replot.sh                       # results/ 안의 가장 최근 폴더 자동 선택

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -ge 1 ]]; then
    RUN_DIR="$1"
else
    RUN_DIR=$(ls -dt results/*/ 2>/dev/null | head -n1)
    if [[ -z "${RUN_DIR}" ]]; then
        echo "ERROR: results/ 아래에 실행 결과가 없습니다." >&2
        exit 1
    fi
    RUN_DIR="${RUN_DIR%/}"
    echo "[auto] 가장 최근 결과 폴더: ${RUN_DIR}"
fi

if [[ ! -d "${RUN_DIR}/embeddings" ]]; then
    echo "ERROR: ${RUN_DIR}/embeddings 디렉토리가 없습니다." >&2
    exit 1
fi

echo "라벨 재시각화 → ${RUN_DIR}"
python scripts/replot_labels.py --run_dir "${RUN_DIR}"

echo
echo "완료. 생성 파일:"
ls -lh "${RUN_DIR}"/umap_by_label_each.png \
       "${RUN_DIR}"/umap_normal_vs_any.png \
       "${RUN_DIR}"/silhouette_per_label.csv \
       "${RUN_DIR}"/silhouette_normal_vs_any.csv 2>/dev/null || true
