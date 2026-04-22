"""
연령대별 UMAP 시각화 (generic)
================================
`run_all_embedding_umap.py` 로 생성된 `results/{tag}/` 결과 디렉토리를 읽어
각 샘플을 연령대 bin으로 나눠 UMAP에 색칠합니다.

config에서 각 데이터셋마다 `age_col`, `age_scale`을 지정하면 해당 컬럼을
사용합니다. 데이터셋 구분은 마커로 표시됩니다.

Config 예시:
  {
    "datasets": [
      {"name": "PTB-XL", "table_csv": "...", "age_col": "age", "age_scale": 100}
    ],
    "age_bins": [
      [0, 3, "#CC79A7"], [3, 6, "#9400D3"], ...
    ]
  }

사용법:
  python scripts/plot_age_umap.py --run_dir results/20260101_120000
  python scripts/plot_age_umap.py --run_dir results/20260101_120000 --balanced
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


# 기본 연령 bin (config에서 override 가능) — (lower, upper, color)
DEFAULT_AGE_BINS = [
    (0,    3,   "#CC79A7"),    # 핑크
    (3,    6,   "#9400D3"),    # 보라
    (6,    12,  "#D55E00"),    # 빨강-주황
    (12,   18,  "#F0E442"),    # 노랑
    (18,   40,  "#0072B2"),    # 파랑
    (40,   60,  "#009E73"),    # 청록
    (60,   80,  "#E69F00"),    # 주황
    (80,   200, "#000000"),    # 검정
]


def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def bin_label(age, bins):
    for lo, hi, _ in bins:
        if lo <= age < hi:
            return f"{lo}-{hi if hi < 200 else '+'}"
    return None


def main():
    parser = argparse.ArgumentParser(description="연령대별 UMAP")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="run_all_embedding_umap.py 결과 디렉토리 (config.json + embeddings/)")
    parser.add_argument("--balanced", action="store_true",
                        help="각 연령 bin을 동일 샘플 수로 맞춤")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    run_dir = Path(args.run_dir).resolve()
    emb_dir = run_dir / "embeddings"
    config_path = run_dir / "config.json"

    if not config_path.exists():
        logging.error(f"config.json이 없습니다: {config_path}")
        return
    with open(config_path) as f:
        cfg = json.load(f)

    datasets_cfg = cfg["datasets"]

    # age bins (config에서 override 가능)
    if "age_bins" in cfg:
        age_bins = [(lo, hi, color) for lo, hi, color in cfg["age_bins"]]
    else:
        age_bins = DEFAULT_AGE_BINS

    # ── 각 데이터셋의 age 벡터 ──
    ds_ages = {}
    ds_bin_labels = {}
    for ds_cfg in datasets_cfg:
        name = ds_cfg["name"]
        age_col = ds_cfg.get("age_col", "age")
        scale = ds_cfg.get("age_scale", 1.0)
        table = pd.read_csv(ds_cfg["table_csv"], low_memory=False)
        if age_col not in table.columns:
            logging.warning(f"  [{name}] age_col='{age_col}' not found, skipping")
            continue
        ages = table[age_col].values * scale
        bin_labels = np.array([bin_label(a, age_bins) for a in ages])
        ds_ages[name] = ages
        ds_bin_labels[name] = bin_labels
        logging.info(f"[{name}] age range: {ages.min():.1f} ~ {ages.max():.1f}")

    if not ds_ages:
        logging.error("age_col이 있는 데이터셋이 없습니다. config에 age_col을 추가하세요.")
        return

    # bin 카운트 테이블
    all_bin_names = [f"{lo}-{hi if hi < 200 else '+'}" for lo, hi, _ in age_bins]
    logging.info(f"\n{'Bin':<8} " + " ".join(f"{n[:10]:>10}" for n in ds_ages))
    for bn in all_bin_names:
        row = f"  {bn:<6}"
        for dname in ds_ages:
            cnt = (ds_bin_labels[dname] == bn).sum()
            row += f" {cnt:>10}"
        logging.info(row)

    # balanced: 각 bin의 (모든 데이터셋 합산) 최소 크기
    total_bin_counts = []
    for bn in all_bin_names:
        total = sum((ds_bin_labels[dn] == bn).sum() for dn in ds_ages)
        total_bin_counts.append(total)
    nonzero = [c for c in total_bin_counts if c > 0]
    min_bin = min(nonzero) if nonzero else 0
    if args.balanced:
        logging.info(f"Balanced mode: n_per_bin = {min_bin}")

    # ── 모델 목록 (임베딩 파일에서 탐지) ──
    meta_files = sorted(emb_dir.glob("*_meta.json"))
    models = []
    for mf in meta_files:
        with open(mf) as f:
            meta = json.load(f)
        models.append({
            "name": meta["model_name"],
            "safe": sanitize(meta["model_name"]),
            "feature_dim": meta["feature_dim"],
            "datasets": meta["datasets"],
        })
    if not models:
        logging.error(f"모델 meta가 없습니다: {emb_dir}")
        return
    logging.info(f"발견된 모델: {[m['name'] for m in models]}")

    # ── UMAP + 시각화 ──
    from umap import UMAP
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_markers = {}
    markers_cycle = ["o", "^", "s", "D", "v", "P", "X", "*"]
    for i, ds_cfg in enumerate(datasets_cfg):
        dataset_markers[ds_cfg["name"]] = markers_cycle[i % len(markers_cycle)]

    color_map = {f"{lo}-{hi if hi < 200 else '+'}": color
                 for lo, hi, color in age_bins}

    rng = np.random.RandomState(42)
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 10 * n_models))
    if n_models == 1:
        axes = [axes]

    for row_idx, minfo in enumerate(models):
        mname = minfo["name"]
        m_safe = minfo["safe"]
        logging.info(f"\n[{row_idx+1}/{n_models}] {mname}...")

        # 데이터셋별 임베딩 + bin 라벨 수집 + (선택) balanced subsample
        sel_emb, sel_bins, sel_ds = [], [], []
        for dname in ds_ages:
            d_safe = sanitize(dname)
            emb_file = emb_dir / f"{m_safe}_{d_safe}.npy"
            if not emb_file.exists():
                logging.warning(f"    {emb_file} 없음, skip")
                continue
            emb = np.load(emb_file)
            bin_lbls = ds_bin_labels[dname][:len(emb)]

            for bn in all_bin_names:
                idx = np.where(bin_lbls == bn)[0]
                if len(idx) == 0:
                    continue
                if args.balanced and min_bin > 0:
                    # 데이터셋별 비례 샘플링
                    total = sum((ds_bin_labels[dn] == bn).sum() for dn in ds_ages)
                    if total > min_bin:
                        share = int(round(min_bin * len(idx) / total))
                        if 0 < share < len(idx):
                            idx = rng.choice(idx, share, replace=False)
                sel_emb.append(emb[idx])
                sel_bins.extend([bn] * len(idx))
                sel_ds.extend([dname] * len(idx))

        if not sel_emb:
            continue
        all_emb = np.concatenate(sel_emb, axis=0)
        sel_bins = np.array(sel_bins)
        sel_ds = np.array(sel_ds)
        logging.info(f"  n={len(all_emb)}, feature_dim={all_emb.shape[1]}")

        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(all_emb)

        suffix = "_balanced" if args.balanced else ""
        np.save(emb_dir / f"{m_safe}_umap_coords_age{suffix}.npy", coords)

        # 플롯 (연령 bin = 색, 데이터셋 = 마커)
        ax = axes[row_idx]
        for bn in all_bin_names:
            for dname in ds_ages:
                mask = (sel_bins == bn) & (sel_ds == dname)
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color_map[bn], marker=dataset_markers[dname],
                    s=14, alpha=0.65, rasterized=True,
                    edgecolors="black", linewidths=0.25,
                    label=f"{dname} {bn} (n={mask.sum()})",
                )
        ax.set_title(
            f"{mname}  (feature_dim={minfo['feature_dim']}, n={len(all_emb)})",
            fontsize=12,
        )
        ax.legend(fontsize=8, markerscale=2.5, loc="best", ncol=2)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    title = "ECG FMs — UMAP by Age Bin (color)"
    if args.balanced:
        title += f" (balanced n_per_bin={min_bin})"
    title += "\n(Marker: " + ", ".join(
        f"{m}={n}" for n, m in dataset_markers.items()
    ) + ")"
    plt.suptitle(title, fontsize=14, y=1.005)
    plt.tight_layout()

    suffix = "_balanced" if args.balanced else ""
    fig_path = run_dir / f"umap_by_age{suffix}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    logging.info(f"\n그림 저장: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
