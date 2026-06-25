# visualize_ecgfm

9개 ECG Foundation 모델의 임베딩을 추출하고 UMAP으로 비교 시각화하는 툴킷입니다.
임의의 H5 ECG 데이터셋에 대해 config 파일 하나만 만들면 동작합니다.

제공 기능:
- **임베딩 추출 + UMAP** : 데이터셋/모델별 임베딩을 뽑고 UMAP으로 비교 (`run_all_embedding_umap.py`)
- **소아 vs 성인 분석** : PTB-XL(성인) / ZZU-pECG(소아) 비교 그리드 (`plot_*`)
- **HEEDB 비지도 클러스터링** : 라벨 무관 표본의 KMeans/HDBSCAN + 대표 파형 (`cluster_heedb.py`)
- **인터랙티브 탐색** : Streamlit 앱 / 노트북 모듈로 프리셋 비교 (`umap_explorer.py`, `umap_view.py`)

## 지원 모델 (9종)

| 모델 | feature_dim | 체크포인트 파일 |
|------|:-----------:|-----------------|
| ECG-JEPA | 768 | `ecg_jepa/multiblock_epoch100.pth` |
| ECG-FM | 768 | `ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` |
| ECG-Founder | 1024 | `ecg_founder/12_lead_ECGFounder.pth` |
| ST-MEM | 768 | `st_mem/st_mem_vit_base_full.pth` |
| MERL (ResNet) | 512 | `merl/res18_best_encoder.pth` |
| MERL (ViT) | 192 | `merl/vit_tiny_best_encoder.pth` |
| ECG-FM-KED | 768 | `ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt` |
| HuBERT-ECG | 768 | `hubert_ecg/hubert_ecg_base.safetensors` |
| CPC | 512 | `cpc/last_11597276.ckpt` |

체크포인트는 config의 `model_dir`를 루트로 위 상대경로에서 로드됩니다.

## 디렉토리 구조

```
visualize_ecgfm/
├── README.md
├── requirements.txt
├── configs/                        # 데이터셋/모델 정의 JSON
│   ├── example.json                #   최소 예시
│   ├── ptbxl_zzu.json              #   PTB-XL + ZZU-pECG (소아/성인)
│   ├── tof_ecg.json                #   ToF-ECG (8→12 lead, nested H5 schema)
│   ├── all_extra.json              #   chapman/sph/cpsc2021/code15/mimic4
│   └── code15_mimic4.json          #   code15 + mimic4 만
├── src/
│   ├── dataset.py                  # H5ECGDataset
│   └── encoders/                   # 9개 인코더 래퍼
├── scripts/                        # 실행 스크립트 (아래 "스크립트" 절 참고)
├── labels/
│   ├── ptbxl_super_bench_labels.csv
│   └── zzu_bench_labels.csv
├── notebooks/                      # 탐색/논문용 노트북 (출력 비움)
└── third_party/                    # 번들된 모델 소스 (clinical_ts, ecg_jepa)
```

## 설치

```bash
git clone https://github.com/tyoon11/visualize_ecgfm.git
cd visualize_ecgfm
pip install -r requirements.txt
```

## Config 작성

`configs/example.json`을 복사해서 본인 데이터셋에 맞게 수정:

```jsonc
{
  "model_dir": "/path/to/ECGFMs",        // 모델 체크포인트 루트
  "models": null,                         // null = 전부, 또는 ["ECG-JEPA", ...]
  "target_fs": 500,                       // 리샘플 후 sampling rate
  "target_length": 5000,                  // 샘플 길이 (10s @ 500Hz)
  "datasets": [
    {
      "name": "MyDataset",
      "h5_root": "/path/to/h5",           // H5 파일이 있는 루트
      "table_csv": "/path/to/table.csv",  // filepath, age 등을 가진 메타 테이블
      "label_csv": "/path/to/labels.csv", // (선택) Normal/Abnormal 라벨 CSV
      "label_col": "NORM",                // (선택) Normal로 간주할 컬럼명
      "age_col": "age",                   // (선택) 연령 UMAP용 컬럼
      "age_scale": 100,                   // (선택) age 원본 × scale
      "display_color": "#1f77b4",         // (선택) 플롯 색
      "display_marker": "o"               // (선택) 플롯 마커
    }
  ],
  "age_bins": [                           // (선택) 연령 bin + 색 커스터마이즈
    [0, 3, "#CC79A7"],   [3, 6, "#9400D3"],   [6, 12, "#D55E00"],
    [12, 18, "#F0E442"], [18, 40, "#0072B2"], [40, 60, "#009E73"],
    [60, 80, "#E69F00"], [80, 200, "#000000"]
  ]
}
```

데이터셋마다 H5 schema가 다르면 dataset 항목에 `"h5_schema"`(예: `"tof"`)와
join 키(`"join_key"`)를 지정합니다. 동작 예시는 `configs/tof_ecg.json` 참고.

## 워크플로우

### 1) 임베딩 추출 + UMAP (메인 파이프라인)

```bash
# 결과는 results/{timestamp}/ 에 저장
python scripts/run_all_embedding_umap.py \
    --config configs/my_datasets.json \
    --n_samples 0 --batch_size 256 --gpus 0,1
```

주요 옵션:
- `--config PATH` : 데이터셋/모델 config JSON (필수)
- `--n_samples 0` : 전체 샘플 (양수면 해당 수만)
- `--batch_size 256` : per-GPU 배치
- `--num_workers N` : DataLoader worker 수
- `--gpus "0,1"` : 사용할 GPU ID (생략 시 전체)
- `--output_root DIR` : 출력 루트 (기본 `results/`)
- `--tag NAME` : 타임스탬프 대신 원하는 디렉토리명
- `--skip_existing` : 저장된 임베딩 있으면 추출 건너뜀 (UMAP만 재생성)

생성된 임베딩으로 후속 플롯을 다시 그릴 때:

```bash
# 연령대별 UMAP
python scripts/plot_age_umap.py --run_dir results/20260101_120000 --balanced

# 라벨별 / Normal-vs-Any-Abnormal 재시각화
bash scripts/run_replot.sh results/20260101_120000        # 또는 인자 없이 최근 폴더 자동 선택
```

### 2) 소아 vs 성인 분석 (PTB-XL / ZZU-pECG)

`results/embeddings/` 의 임베딩을 재사용해 9개 모델을 한 그리드로 비교합니다.

```bash
# 4-패널 그리드: Adult/Pediatric · Normal/Abnormal · Adult N/A · Pediatric N/A
bash scripts/run_age_split.sh                              # 또는 <EMBEDDINGS_DIR> 지정

# 개별 그리드
python scripts/plot_umap_4panel.py     --embeddings_dir results/embeddings --output_dir results/out
python scripts/plot_normal_abnormal.py --embeddings_dir results/embeddings --output_dir results/out --split_age 18
python scripts/plot_age_umap_split.py  --embeddings_dir results/embeddings --output_dir results/out --split_age 18
```

### 3) HEEDB 비지도 클러스터링

라벨 무관 랜덤 표본의 임베딩을 추출해 KMeans(스윕) + HDBSCAN 클러스터링 후
UMAP으로 시각화하고, 대표 파형(medoid)을 그립니다.

```bash
# 클러스터링 (임베딩+좌표를 results/embeddings/ 에 npz 캐시)
CUDA_VISIBLE_DEVICES=6 python scripts/cluster_heedb.py --encoder founder --n_samples 100000

# 클러스터별 대표 12-lead 파형
CUDA_VISIBLE_DEVICES="" python scripts/plot_cluster_waveforms.py \
    --cluster_ts 20260619_160046 --n_samples 100000 --seed 42

# HEEDB Normal vs Abnormal UMAP / age 재채색
python scripts/umap_heedb_founder.py      --encoder founder --n_per_class 10000
python scripts/umap_heedb_age_recolor.py  --npz results/embeddings/heedb_umap_founder_n10000.npz
```

> 대용량 UMAP은 `NUMBA_NUM_THREADS`를 제한하지 않으면 segfault가 날 수 있습니다.

### 4) 인터랙티브 탐색

```bash
# (선택) UMAP 좌표 캐시 사전 생성 — explorer/view가 즉시 로드
python scripts/precompute_umap_cache.py --emb_dir results/embeddings

# Streamlit 앱: 프리셋(orig / cosineL2 등)을 탭으로 비교
streamlit run scripts/umap_explorer.py -- --result_dir results/embeddings
```

노트북에서는 `umap_view.Explorer`로 streamlit 없이 동일 캐시를 사용:

```python
from scripts.umap_view import Explorer
ex = Explorer("results/embeddings")
ex.print_status()
ex.view(color_mode="dataset")
ex.compare(presets=["orig", "cosineL2"])
ex.view(color_mode="age", age_bins="0,18,40,60,80,200")
```

## 스크립트

| 스크립트 | 설명 |
|----------|------|
| `run_all_embedding_umap.py` | **메인** — 임베딩 추출 + UMAP (모든 모델 × 데이터셋) |
| `plot_age_umap.py` | 연령대별 UMAP (run_dir 재사용) |
| `replot_labels.py` / `run_replot.sh` | 라벨별 · Normal-vs-Any-Abnormal 재시각화 |
| `plot_umap_4panel.py` / `run_age_split.sh` | 소아/성인 4-패널 통합 그리드 |
| `plot_normal_abnormal.py` | Normal vs Abnormal (9 모델 × 3 그룹) |
| `plot_age_umap_split.py` | 소아/성인 분리 UMAP (9 모델 × 2 그룹) |
| `cluster_heedb.py` | HEEDB 비지도 클러스터링 (KMeans/HDBSCAN + UMAP) |
| `plot_cluster_waveforms.py` | 클러스터 대표 파형(medoid) 플롯 |
| `umap_heedb_founder.py` | HEEDB Normal/Abnormal UMAP (인코더별) |
| `umap_heedb_age_recolor.py` | 기존 HEEDB 좌표를 age 기준 재채색 |
| `precompute_umap_cache.py` | (모델 × 프리셋) UMAP 좌표 캐시 사전 생성 |
| `umap_explorer.py` | Streamlit 인터랙티브 UMAP 탐색기 |
| `umap_view.py` | 노트북용 UMAP 뷰 모듈 (`Explorer`) |
| `embedding_analysis.py` | 단일 인코더 t-SNE/UMAP 분석 |
| `build_benchmark_labels.py` | (선택) PTB-XL super / ZZU 벤치 라벨 생성 |

## 출력물 (`results/{timestamp}/`)

```
results/
├── embeddings/                            # 모델/데이터셋 공유 캐시 (임베딩 + UMAP 좌표)
│   ├── {model}_{dataset}.npy
│   ├── {model}_{dataset}_labels.npy
│   ├── {model}_umap_coords_*.npy
│   └── {model}_meta.json
└── 20260101_120000/
    ├── config.json                        # 실행 시 config 백업
    ├── umap_by_dataset.png                # 데이터셋별 색상 UMAP
    ├── umap_by_label.png                  # Normal/Abnormal (라벨 있을 때)
    ├── umap_by_age[_balanced].png         # 연령대별 UMAP
    └── silhouette_scores.csv              # 실루엣 점수
```

> `results/` 는 통째로 `.gitignore` 됩니다 (타임스탬프 단위로 생성/대용량).

## 주의사항

1. **라벨 CSV 포맷** : `filepath` 컬럼 + 각 라벨을 True/False 또는 0/1로 기록
   (dataset의 `table_csv`와 `filepath`로 join됨)
2. **age 값** : 저장 포맷에 따라 `age_scale` 조정 (예: 0~1 정규화면 100)
3. **MERL (ViT)** 제외하려면 config의 `models`에 나머지 8개만 나열
4. **CPC** : Lightning checkpoint의 pickle 메타는 stub 모듈로 자동 처리.
   S4 predictor는 pykeops/CUDA가 없으면 encoder-only fallback
5. **ECG-FM-KED** : fastai v1 → v2 호환 shim을 래퍼 상단에서 자동 설치
6. **대용량 UMAP** : `NUMBA_NUM_THREADS` 제한 없이 큰 표본을 돌리면 segfault 가능
