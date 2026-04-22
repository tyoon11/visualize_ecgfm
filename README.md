# visualize_ecgfm

9개 ECG Foundation 모델의 임베딩을 추출하고 UMAP으로 비교 시각화합니다.
임의의 H5 ECG 데이터셋에 대해 config 파일 하나만 만들면 동작합니다.

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

## 디렉토리 구조

```
visualize_ecgfm/
├── README.md
├── requirements.txt
├── configs/
│   ├── example.json           # 최소 예시
│   └── ptbxl_zzu.json         # PTB-XL + ZZU-pECG
├── src/
│   ├── dataset.py             # H5ECGDataset
│   └── encoders/              # 9개 인코더 래퍼
├── scripts/
│   ├── run_all_embedding_umap.py   # 임베딩 + UMAP
│   ├── plot_age_umap.py            # 연령대별 UMAP (run_dir 재사용)
│   └── build_benchmark_labels.py   # (선택) PTB-XL super 라벨 생성
├── labels/
│   ├── ptbxl_super_bench_labels.csv
│   └── zzu_bench_labels.csv
└── third_party/               # 번들된 모델 소스 (clinical_ts, ecg_jepa)
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
    [0, 3, "#CC79A7"],
    [3, 6, "#9400D3"],
    [6, 12, "#D55E00"],
    [12, 18, "#F0E442"],
    [18, 40, "#0072B2"],
    [40, 60, "#009E73"],
    [60, 80, "#E69F00"],
    [80, 200, "#000000"]
  ]
}
```

## 실행

```bash
# 1) 임베딩 추출 + UMAP (결과는 results/{timestamp}/ 에 저장)
python scripts/run_all_embedding_umap.py \
    --config configs/my_datasets.json \
    --n_samples 0 --batch_size 256 --gpus 0,1

# 2) (선택) 연령대별 UMAP — 1) 실행 결과 디렉토리를 넘김
python scripts/plot_age_umap.py --run_dir results/20260101_120000 --balanced
```

**주요 옵션 (run_all_embedding_umap.py):**
- `--config PATH`: 데이터셋/모델 config JSON (필수)
- `--n_samples 0`: 전체 샘플 (양수면 해당 수만)
- `--batch_size 256`: per-GPU 배치
- `--gpus "0,1"`: 사용할 GPU ID. 생략 시 전체
- `--tag NAME`: 타임스탬프 대신 원하는 이름으로 출력 디렉토리명 지정
- `--skip_existing`: 저장된 임베딩 있으면 추출 건너뜀 (UMAP만 재생성)

## 출력물 (`results/{timestamp}/`)

```
results/
└── 20260101_120000/
    ├── config.json                    # 실행 시 config 백업
    ├── umap_by_dataset.png            # 데이터셋별 색상 UMAP
    ├── umap_by_label.png              # per-dataset Normal/Abnormal (라벨 있을 때)
    ├── umap_by_age[_balanced].png     # 연령대별 UMAP
    ├── silhouette_scores.csv          # 실루엣 점수
    └── embeddings/
        ├── {model}_{dataset}.npy           # 임베딩
        ├── {model}_{dataset}_labels.npy    # multi-hot 라벨
        ├── {model}_umap_coords_combined.npy
        ├── {model}_umap_coords_age[_balanced].npy
        └── {model}_meta.json
```

## 주의사항

1. **라벨 CSV 포맷**: `filepath` 컬럼 + 각 라벨을 True/False 또는 0/1로 기록
   (dataset의 `table_csv`와 `filepath`로 join됨)
2. **age 값**: 저장 포맷에 따라 `age_scale` 조정 (예: 0~1 정규화면 100)
3. **MERL (ViT)** 제외하려면 config의 `models`에 나머지 8개만 나열
4. **CPC**: Lightning checkpoint의 pickle 메타는 stub 모듈로 자동 처리. S4 predictor는 pykeops/CUDA가 없으면 encoder-only fallback
5. **ECG-FM-KED**: fastai v1 → v2 호환 shim을 래퍼 상단에서 자동 설치
