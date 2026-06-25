"""
Normal vs Abnormal UMAP — 9 모델 × 3 그룹 통합 그리드
=======================================================
각 (모델, 그룹) 셀에 대해 UMAP을 따로 fit (좌표 캐시 재사용).
Normal(초록) vs Abnormal(빨강) 색상, 데이터셋(PTB-XL=o / ZZU=^) 마커.

입력: `results/embeddings/`
출력: `{output_dir}/umap_normal_abnormal.png` (9 × 3 그리드)

그룹: combined (전체), adult (≥split_age), pediatric (<split_age)
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_EMB_DIR = str(SCRIPT_DIR / "results" / "embeddings")
PTBXL_TABLE = "/home/irteam/ddn-opendata1/h5/physionet/v2.0/ptbxl_table.csv"
ZZU_TABLE = "/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0/ecg_table.csv"

MODEL_ORDER = [
    "CPC",
    "ECG-FM-KED",
    "ECG-FM",
    "ECG-Founder",
    "ECG-JEPA",
    "HuBERT-ECG",
    "MERL (ResNet)",
    "MERL (ViT)",
    "ST-MEM",
]

COLOR_NORMAL = "#2ecc71"
COLOR_ABNORMAL = "#e74c3c"


def sanitize(name: str) -> str:
    return (name.replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace("/", "_"))


def valid(a):
    return np.isfinite(a) & (a >= 0)


def load_model_list(emb_dir: Path):
    models = []
    for display in MODEL_ORDER:
        safe = sanitize(display)
        if (emb_dir / f"{safe}_ptbxl.npy").exists() \
                and (emb_dir / f"{safe}_zzu.npy").exists():
            models.append({"safe": safe, "display": display})
    return models


def feature_dim_of(emb_dir: Path, safe: str, fallback: int) -> int:
    p = emb_dir / f"{safe}_meta.npz"
    if p.exists():
        try:
            return int(np.load(p, allow_pickle=True)["feature_dim"])
        except Exception:
            pass
    return fallback


def fit_or_load_umap(embs, cache_path: Path, umap_cls):
    if cache_path.exists():
        coords = np.load(cache_path)
        if len(coords) == len(embs) and coords.shape[1] == 2:
            return coords, True
    reducer = umap_cls(n_components=2, random_state=42,
                       n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embs)
    np.save(cache_path, coords)
    return coords, False


def balance_classes(is_norm, rng):
    """Normal/Abnormal을 1:1로 다운샘플링한 인덱스 반환."""
    norm_idx = np.where(is_norm)[0]
    abn_idx = np.where(~is_norm)[0]
    n_each = min(len(norm_idx), len(abn_idx))
    if n_each == 0:
        return None
    sel_n = rng.choice(norm_idx, n_each, replace=False) if len(norm_idx) > n_each else norm_idx
    sel_a = rng.choice(abn_idx, n_each, replace=False) if len(abn_idx) > n_each else abn_idx
    keep = np.concatenate([sel_n, sel_a])
    keep.sort()
    return keep


def main():
    parser = argparse.ArgumentParser(
        description="Normal vs Abnormal UMAP 통합 그리드"
    )
    parser.add_argument("--embeddings_dir", type=str, default=DEFAULT_EMB_DIR)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split_age", type=float, default=18.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    emb_dir = Path(args.embeddings_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else emb_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ptbxl_ages = pd.to_numeric(
        pd.read_csv(PTBXL_TABLE, low_memory=False)["age"], errors="coerce"
    ).values * 100.0
    zzu_ages = pd.to_numeric(
        pd.read_csv(ZZU_TABLE, low_memory=False)["age"], errors="coerce"
    ).values * 100.0

    models = load_model_list(emb_dir)
    if not models:
        logging.error(f"임베딩 없음: {emb_dir}")
        return
    n_models = len(models)
    logging.info(f"모델 {n_models}: {[m['display'] for m in models]}")

    from umap import UMAP
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.metrics import silhouette_score

    split = args.split_age
    groups = [
        ("combined",  lambda a: np.ones_like(a, dtype=bool)),
        ("adult",     lambda a: a >= split),
        ("pediatric", lambda a: a < split),
    ]

    fig, axes = plt.subplots(n_models, 3, figsize=(30, 8 * n_models))
    if n_models == 1:
        axes = axes[None, :]

    for col_idx, (gname, mask_fn) in enumerate(groups):
        for row_idx, minfo in enumerate(models):
            safe = minfo["safe"]
            display = minfo["display"]

            emb_p = np.load(emb_dir / f"{safe}_ptbxl.npy")
            emb_z = np.load(emb_dir / f"{safe}_zzu.npy")
            lbl_p = np.load(emb_dir / f"{safe}_ptbxl_labels.npy")
            lbl_z = np.load(emb_dir / f"{safe}_zzu_labels.npy")

            n_p = min(len(emb_p), len(ptbxl_ages), len(lbl_p))
            n_z = min(len(emb_z), len(zzu_ages), len(lbl_z))
            age_p = ptbxl_ages[:n_p]
            age_z = zzu_ages[:n_z]
            norm_p = (lbl_p[:n_p, 0] > 0).astype(bool)
            norm_z = (lbl_z[:n_z, 0] > 0).astype(bool)

            mask_p = mask_fn(age_p) & valid(age_p)
            mask_z = mask_fn(age_z) & valid(age_z)
            if mask_p.sum() + mask_z.sum() == 0:
                axes[row_idx, col_idx].set_visible(False)
                continue

            embs = np.concatenate([emb_p[:n_p][mask_p], emb_z[:n_z][mask_z]], axis=0)
            is_norm = np.concatenate([norm_p[mask_p], norm_z[mask_z]])
            ds = np.array(
                ["ptbxl"] * int(mask_p.sum()) + ["zzu"] * int(mask_z.sum())
            )

            # Normal/Abnormal 1:1 다운샘플
            rng = np.random.RandomState(42)
            keep = balance_classes(is_norm, rng)
            if keep is None:
                axes[row_idx, col_idx].set_visible(False)
                logging.warning(f"  [{gname}] {display}: 한쪽 클래스 0개, skip")
                continue
            embs = embs[keep]
            is_norm = is_norm[keep]
            ds = ds[keep]

            cache = emb_dir / f"{safe}_umap_coords_na_{gname}_balanced.npy"
            coords, cached = fit_or_load_umap(embs, cache, UMAP)

            y = is_norm.astype(int)
            sil = (silhouette_score(embs, y, sample_size=min(10000, len(embs)))
                   if len(set(y)) > 1 else float("nan"))

            logging.info(
                f"[{gname}][{row_idx+1}/{n_models}] {display}: n={len(embs)} "
                f"(N=Ab={int(is_norm.sum())} each, "
                f"ptbxl={(ds=='ptbxl').sum()}, zzu={(ds=='zzu').sum()}), "
                f"sil={sil:.3f}" + ("  [cached]" if cached else "")
            )

            ax = axes[row_idx, col_idx]
            for (lbl_val, color) in [(True, COLOR_NORMAL), (False, COLOR_ABNORMAL)]:
                for ds_key, mk in [("ptbxl", "o"), ("zzu", "^")]:
                    m = (is_norm == lbl_val) & (ds == ds_key)
                    if m.sum() == 0:
                        continue
                    ax.scatter(
                        coords[m, 0], coords[m, 1],
                        c=color, marker=mk,
                        s=10, alpha=0.55, rasterized=True,
                        edgecolors="black", linewidths=0.12,
                    )

            feat = feature_dim_of(emb_dir, safe, embs.shape[1])
            title_bits = []
            if row_idx == 0:
                title_bits.append(f"[{gname.upper()}]  ")
            title_bits.append(f"{display}")
            title_bits.append(
                f"(feat={feat}, balanced n={len(embs)}, sil={sil:.3f})"
            )
            ax.set_title("  ".join(title_bits), fontsize=11)
            ax.set_xlabel("UMAP 1")
            if col_idx == 0:
                ax.set_ylabel("UMAP 2")

            legend_elems = [
                Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=COLOR_NORMAL, markersize=8, label="Normal"),
                Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=COLOR_ABNORMAL, markersize=8, label="Abnormal"),
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="gray", markeredgecolor="black",
                       markersize=7, label="PTB-XL"),
                Line2D([0], [0], marker="^", color="w",
                       markerfacecolor="gray", markeredgecolor="black",
                       markersize=7, label="ZZU-pECG"),
            ]
            ax.legend(handles=legend_elems, loc="best", fontsize=8, ncol=2)

    plt.suptitle(
        "ECG FMs — Normal vs Abnormal UMAP (Normal:Abnormal balanced 1:1 per cell)   "
        "(columns: combined / adult / pediatric)   "
        "Normal=Green, Abnormal=Red   Marker: ●PTB-XL ▲ZZU",
        fontsize=13, y=1.001,
    )
    plt.tight_layout()
    fig_path = out_dir / "umap_normal_abnormal.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"저장: {fig_path}")


if __name__ == "__main__":
    main()
