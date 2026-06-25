"""
소아/성인 분리 UMAP (9 모델 × 2 그룹 통합 그리드)
=====================================================
입력: `results/embeddings/`
  - {safe}_ptbxl.npy,        {safe}_zzu.npy              : 임베딩
  - {safe}_ptbxl_labels.npy, {safe}_zzu_labels.npy       : multi-hot (idx 0 = normal)
  - {safe}_meta.npz                                       : feature_dim

동작:
  - age < split_age → pediatric, 그 외 → adult
  - 각 (모델, 그룹) 셀에 대해 UMAP을 따로 fit (기존 좌표 캐시가 있으면 재사용)
  - 하나의 그리드 (n_models × 2 컬럼)로 저장: `umap_by_age.png`
  - 색: Normal=Blues / Abnormal=Reds, 농도=그룹 내 age min-max 정규화
  - 마커: PTB-XL=o, ZZU=^
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

GROUPS = [
    ("pediatric", lambda a: a < 18.0),   # split_age가 기본 18일 때 사용될 lambda
    ("adult",     lambda a: a >= 18.0),
]


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
            return coords
    reducer = umap_cls(n_components=2, random_state=42,
                       n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embs)
    np.save(cache_path, coords)
    return coords


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
        description="소아/성인 분리 UMAP 통합 그리드 (연령 그라데이션)"
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
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cmap_normal = plt.get_cmap("Blues")
    cmap_abnormal = plt.get_cmap("Reds")

    def to_rgba(ages, is_norm, norm):
        frac = 0.3 + 0.65 * norm(ages)
        frac = np.clip(frac, 0.0, 1.0)
        rgba = np.empty((len(ages), 4))
        rgba[is_norm] = cmap_normal(frac[is_norm])
        rgba[~is_norm] = cmap_abnormal(frac[~is_norm])
        return rgba

    # mask functions (split_age 적용)
    split = args.split_age
    groups = [
        ("pediatric", lambda a: a < split),
        ("adult",     lambda a: a >= split),
    ]

    # 컬럼별 공통 age 범위 (colormap 정규화)
    col_norms = {}
    for gname, mask_fn in groups:
        pooled = np.concatenate([
            ptbxl_ages[mask_fn(ptbxl_ages) & valid(ptbxl_ages)],
            zzu_ages[mask_fn(zzu_ages) & valid(zzu_ages)],
        ])
        if len(pooled) == 0:
            col_norms[gname] = (0.0, 1.0, Normalize(0, 1))
            continue
        lo, hi = float(pooled.min()), float(pooled.max())
        if hi - lo < 1e-6:
            hi = lo + 1.0
        col_norms[gname] = (lo, hi, Normalize(vmin=lo, vmax=hi))
        logging.info(f"[{gname}] age 범위: {lo:.1f}~{hi:.1f} (n_pool={len(pooled)})")

    fig, axes = plt.subplots(n_models, 2, figsize=(22, 8.5 * n_models))
    if n_models == 1:
        axes = axes[None, :]

    for col_idx, (gname, mask_fn) in enumerate(groups):
        age_lo, age_hi, norm = col_norms[gname]

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
            ages = np.concatenate([age_p[mask_p], age_z[mask_z]])
            is_norm = np.concatenate([norm_p[mask_p], norm_z[mask_z]])
            ds = np.array(
                ["ptbxl"] * int(mask_p.sum()) + ["zzu"] * int(mask_z.sum())
            )

            # Normal/Abnormal 1:1 다운샘플 (셀별 random_state 동일)
            rng = np.random.RandomState(42)
            keep = balance_classes(is_norm, rng)
            if keep is None:
                axes[row_idx, col_idx].set_visible(False)
                logging.warning(f"  [{gname}] {display}: 한쪽 클래스 0개, skip")
                continue
            embs = embs[keep]
            ages = ages[keep]
            is_norm = is_norm[keep]
            ds = ds[keep]

            cache = emb_dir / f"{safe}_umap_coords_age_{gname}_balanced.npy"
            coords = fit_or_load_umap(embs, cache, UMAP)
            logging.info(
                f"[{gname}][{row_idx+1}/{n_models}] {display}: n={len(embs)} "
                f"(N=Ab={int(is_norm.sum())} each, ptbxl={(ds=='ptbxl').sum()}, "
                f"zzu={(ds=='zzu').sum()})"
                + ("  [cached]" if cache.exists() else "")
            )

            ax = axes[row_idx, col_idx]
            rgba = to_rgba(ages, is_norm, norm)
            for ds_key, mk in [("ptbxl", "o"), ("zzu", "^")]:
                m = ds == ds_key
                if m.sum() == 0:
                    continue
                ax.scatter(
                    coords[m, 0], coords[m, 1],
                    c=rgba[m], marker=mk,
                    s=14, alpha=0.72, rasterized=True,
                    edgecolors="black", linewidths=0.15,
                )

            feat = feature_dim_of(emb_dir, safe, embs.shape[1])
            title_bits = [f"{display}"]
            if row_idx == 0:
                title_bits.insert(0, f"[{gname.upper()}]  ")
            title_bits.append(
                f"(feat={feat}, n={len(embs)}, balanced N=Ab={int(is_norm.sum())})"
            )
            ax.set_title("  ".join(title_bits), fontsize=11)
            ax.set_xlabel("UMAP 1")
            if col_idx == 0:
                ax.set_ylabel("UMAP 2")

            # colorbars (각 셀마다)
            sm_n = ScalarMappable(norm=norm, cmap=cmap_normal); sm_n.set_array([])
            sm_a = ScalarMappable(norm=norm, cmap=cmap_abnormal); sm_a.set_array([])
            fig.colorbar(sm_n, ax=ax, fraction=0.03, pad=0.01).set_label(
                "Normal age", fontsize=8
            )
            fig.colorbar(sm_a, ax=ax, fraction=0.03, pad=0.02).set_label(
                "Abnormal age", fontsize=8
            )

            legend_elems = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="gray", markeredgecolor="black",
                       markersize=7, label="PTB-XL"),
                Line2D([0], [0], marker="^", color="w",
                       markerfacecolor="gray", markeredgecolor="black",
                       markersize=7, label="ZZU-pECG"),
            ]
            ax.legend(handles=legend_elems, loc="best", fontsize=8)

    ped_lo, ped_hi, _ = col_norms["pediatric"]
    adu_lo, adu_hi, _ = col_norms["adult"]
    plt.suptitle(
        f"ECG FMs — UMAP by Age Group (Normal:Abnormal balanced 1:1 per cell)   "
        f"(pediatric: {ped_lo:.0f}–{ped_hi:.0f}, adult: {adu_lo:.0f}–{adu_hi:.0f})   "
        f"Blue=Normal / Red=Abnormal, intensity=age",
        fontsize=13, y=1.001,
    )
    plt.tight_layout()
    fig_path = out_dir / "umap_by_age.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"저장: {fig_path}")


if __name__ == "__main__":
    main()
