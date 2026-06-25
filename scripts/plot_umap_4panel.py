"""
4-Panel UMAP (PTB-XL=Adult, ZZU-pECG=Pediatric) — 9 모델 × 4 컬럼 통합 그리드
==============================================================================
각 셀에서 1:1 비율로 다운샘플링한 후 UMAP을 따로 fit한다.

Columns
  0  Adult vs Pediatric   : PTB-XL(=Adult) vs ZZU-pECG(=Pediatric)
                            balance: n_PTB-XL = n_ZZU
                            color : Adult=steelblue, Pediatric=coral
  1  Normal vs Abnormal   : combined (PTB-XL + ZZU)
                            balance: n_Normal = n_Abnormal (combined pool)
                            color : Normal=#2ecc71, Abnormal=#e74c3c
                            marker: PTB-XL=o, ZZU=^
  2  Adult N vs A         : PTB-XL only
                            balance: n_NORM = n_~NORM
  3  Pediatric N vs A     : ZZU only
                            balance: n_isNormal = n_~isNormal

입력 : results/embeddings/  ({safe}_ptbxl.npy, _zzu.npy, _ptbxl_labels.npy, _zzu_labels.npy)
출력 : <output_dir>/umap_4panel.png
캐시 : results/embeddings/{safe}_umap_coords_<panel_key>_balanced.npy
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_EMB_DIR = str(SCRIPT_DIR / "results" / "embeddings")

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

COLOR_ADULT = "#4682b4"   # steelblue
COLOR_PEDI = "#ff7f50"    # coral
COLOR_NORMAL = "#2ecc71"
COLOR_ABNORMAL = "#e74c3c"
MARKER_PTBXL = "o"
MARKER_ZZU = "^"

PANEL_KEYS = ["adult_vs_ped", "normal_vs_ab", "adult_normal_vs_ab", "pedi_normal_vs_ab"]
PANEL_TITLES = [
    "Adult vs Pediatric",
    "Normal vs Abnormal (combined)",
    "Adult: Normal vs Abnormal",
    "Pediatric: Normal vs Abnormal",
]


def sanitize(name: str) -> str:
    return (name.replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace("/", "_"))


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


def balance_two(idx_a, idx_b, rng):
    """두 인덱스 그룹을 1:1 다운샘플 → (sel_a, sel_b)."""
    n = min(len(idx_a), len(idx_b))
    if n == 0:
        return None, None
    sa = rng.choice(idx_a, n, replace=False) if len(idx_a) > n else np.asarray(idx_a)
    sb = rng.choice(idx_b, n, replace=False) if len(idx_b) > n else np.asarray(idx_b)
    return sa, sb


def main():
    parser = argparse.ArgumentParser(
        description="4-Panel UMAP (Adult vs Ped / N vs A / Adult N-A / Ped N-A)"
    )
    parser.add_argument("--embeddings_dir", type=str, default=DEFAULT_EMB_DIR)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="기본: embeddings_dir의 부모 폴더")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    emb_dir = Path(args.embeddings_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else emb_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

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

    fig, axes = plt.subplots(n_models, 4, figsize=(28, 7 * n_models))
    if n_models == 1:
        axes = axes[None, :]

    rng_seed = 42

    for row_idx, minfo in enumerate(models):
        safe = minfo["safe"]
        display = minfo["display"]
        feat = feature_dim_of(emb_dir, safe, 0)

        emb_p = np.load(emb_dir / f"{safe}_ptbxl.npy")
        emb_z = np.load(emb_dir / f"{safe}_zzu.npy")
        lbl_p = np.load(emb_dir / f"{safe}_ptbxl_labels.npy")
        lbl_z = np.load(emb_dir / f"{safe}_zzu_labels.npy")

        # idx 0: PTB-XL NORM, ZZU is_Normal (True=normal)
        norm_p = (lbl_p[:, 0] > 0).astype(bool)
        norm_z = (lbl_z[:, 0] > 0).astype(bool)

        # ── Panel 0: Adult (PTB-XL) vs Pediatric (ZZU), 1:1 ──
        rng = np.random.RandomState(rng_seed)
        idx_p_all = np.arange(len(emb_p))
        idx_z_all = np.arange(len(emb_z))
        sel_p, sel_z = balance_two(idx_p_all, idx_z_all, rng)
        embs0 = np.concatenate([emb_p[sel_p], emb_z[sel_z]], axis=0)
        ds0 = np.array(["ptbxl"] * len(sel_p) + ["zzu"] * len(sel_z))
        cache0 = emb_dir / f"{safe}_umap_coords_{PANEL_KEYS[0]}_balanced.npy"
        coords0, c0 = fit_or_load_umap(embs0, cache0, UMAP)
        sil0 = silhouette_score(
            embs0, (ds0 == "zzu").astype(int),
            sample_size=min(10000, len(embs0)),
        )
        logging.info(
            f"[{row_idx+1}/{n_models}] {display} P0 Adult/Ped: "
            f"n={len(embs0)} ({len(sel_p)} each), sil={sil0:.3f}"
            + ("  [cached]" if c0 else "")
        )

        ax = axes[row_idx, 0]
        m = ds0 == "ptbxl"
        ax.scatter(coords0[m, 0], coords0[m, 1], c=COLOR_ADULT,
                   marker="o", s=10, alpha=0.5, rasterized=True,
                   edgecolors="black", linewidths=0.1)
        m = ds0 == "zzu"
        ax.scatter(coords0[m, 0], coords0[m, 1], c=COLOR_PEDI,
                   marker="o", s=10, alpha=0.5, rasterized=True,
                   edgecolors="black", linewidths=0.1)
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_ADULT,
                   markersize=8, label=f"PTB-XL (Adult, n={len(sel_p)})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PEDI,
                   markersize=8, label=f"ZZU-pECG (Ped, n={len(sel_z)})"),
        ], loc="best", fontsize=9)

        # ── Panel 1: Normal vs Abnormal (combined), 1:1 ──
        rng = np.random.RandomState(rng_seed + 1)
        norm_pool_idx = np.concatenate([
            np.where(norm_p)[0],
            len(emb_p) + np.where(norm_z)[0],
        ])
        abn_pool_idx = np.concatenate([
            np.where(~norm_p)[0],
            len(emb_p) + np.where(~norm_z)[0],
        ])
        sel_n, sel_a = balance_two(norm_pool_idx, abn_pool_idx, rng)
        all_emb = np.concatenate([emb_p, emb_z], axis=0)
        keep = np.concatenate([sel_n, sel_a])
        embs1 = all_emb[keep]
        is_norm1 = np.concatenate([np.ones(len(sel_n), bool), np.zeros(len(sel_a), bool)])
        ds1 = np.array(
            ["ptbxl" if i < len(emb_p) else "zzu" for i in keep]
        )
        cache1 = emb_dir / f"{safe}_umap_coords_{PANEL_KEYS[1]}_balanced.npy"
        coords1, c1 = fit_or_load_umap(embs1, cache1, UMAP)
        sil1 = silhouette_score(
            embs1, is_norm1.astype(int),
            sample_size=min(10000, len(embs1)),
        )
        logging.info(
            f"           P1 N/A combined: n={len(embs1)} "
            f"({len(sel_n)} each, ptbxl={(ds1=='ptbxl').sum()}, zzu={(ds1=='zzu').sum()}), "
            f"sil={sil1:.3f}" + ("  [cached]" if c1 else "")
        )

        ax = axes[row_idx, 1]
        for is_n, color in [(True, COLOR_NORMAL), (False, COLOR_ABNORMAL)]:
            for ds_key, mk in [("ptbxl", MARKER_PTBXL), ("zzu", MARKER_ZZU)]:
                m = (is_norm1 == is_n) & (ds1 == ds_key)
                if m.sum() == 0:
                    continue
                ax.scatter(coords1[m, 0], coords1[m, 1], c=color,
                           marker=mk, s=10, alpha=0.5, rasterized=True,
                           edgecolors="black", linewidths=0.1)
        ax.legend(handles=[
            Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_NORMAL,
                   markersize=8, label=f"Normal (n={len(sel_n)})"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_ABNORMAL,
                   markersize=8, label=f"Abnormal (n={len(sel_a)})"),
            Line2D([0], [0], marker=MARKER_PTBXL, color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=7, label="PTB-XL"),
            Line2D([0], [0], marker=MARKER_ZZU, color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=7, label="ZZU-pECG"),
        ], loc="best", fontsize=8, ncol=2)

        # ── Panel 2: PTB-XL only Normal vs Abnormal, 1:1 ──
        rng = np.random.RandomState(rng_seed + 2)
        sel_n2, sel_a2 = balance_two(np.where(norm_p)[0], np.where(~norm_p)[0], rng)
        keep2 = np.concatenate([sel_n2, sel_a2])
        embs2 = emb_p[keep2]
        is_norm2 = np.concatenate([np.ones(len(sel_n2), bool), np.zeros(len(sel_a2), bool)])
        cache2 = emb_dir / f"{safe}_umap_coords_{PANEL_KEYS[2]}_balanced.npy"
        coords2, c2 = fit_or_load_umap(embs2, cache2, UMAP)
        sil2 = silhouette_score(
            embs2, is_norm2.astype(int),
            sample_size=min(10000, len(embs2)),
        )
        logging.info(
            f"           P2 Adult N/A:    n={len(embs2)} ({len(sel_n2)} each), "
            f"sil={sil2:.3f}" + ("  [cached]" if c2 else "")
        )

        ax = axes[row_idx, 2]
        for is_n, color in [(True, COLOR_NORMAL), (False, COLOR_ABNORMAL)]:
            m = is_norm2 == is_n
            ax.scatter(coords2[m, 0], coords2[m, 1], c=color,
                       marker=MARKER_PTBXL, s=10, alpha=0.5, rasterized=True,
                       edgecolors="black", linewidths=0.1)
        ax.legend(handles=[
            Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_NORMAL,
                   markersize=8, label=f"Normal (n={len(sel_n2)})"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_ABNORMAL,
                   markersize=8, label=f"Abnormal (n={len(sel_a2)})"),
        ], loc="best", fontsize=9)

        # ── Panel 3: ZZU only Normal vs Abnormal, 1:1 ──
        rng = np.random.RandomState(rng_seed + 3)
        sel_n3, sel_a3 = balance_two(np.where(norm_z)[0], np.where(~norm_z)[0], rng)
        keep3 = np.concatenate([sel_n3, sel_a3])
        embs3 = emb_z[keep3]
        is_norm3 = np.concatenate([np.ones(len(sel_n3), bool), np.zeros(len(sel_a3), bool)])
        cache3 = emb_dir / f"{safe}_umap_coords_{PANEL_KEYS[3]}_balanced.npy"
        coords3, c3 = fit_or_load_umap(embs3, cache3, UMAP)
        sil3 = silhouette_score(
            embs3, is_norm3.astype(int),
            sample_size=min(10000, len(embs3)),
        )
        logging.info(
            f"           P3 Ped N/A:      n={len(embs3)} ({len(sel_n3)} each), "
            f"sil={sil3:.3f}" + ("  [cached]" if c3 else "")
        )

        ax = axes[row_idx, 3]
        for is_n, color in [(True, COLOR_NORMAL), (False, COLOR_ABNORMAL)]:
            m = is_norm3 == is_n
            ax.scatter(coords3[m, 0], coords3[m, 1], c=color,
                       marker=MARKER_ZZU, s=10, alpha=0.5, rasterized=True,
                       edgecolors="black", linewidths=0.1)
        ax.legend(handles=[
            Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_NORMAL,
                   markersize=8, label=f"Normal (n={len(sel_n3)})"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_ABNORMAL,
                   markersize=8, label=f"Abnormal (n={len(sel_a3)})"),
        ], loc="best", fontsize=9)

        # ── 컬럼/모델 타이틀 ──
        sils = [sil0, sil1, sil2, sil3]
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            head = f"[{PANEL_TITLES[col_idx]}]\n" if row_idx == 0 else ""
            ax.set_title(
                f"{head}{display}  (feat={feat}, sil={sils[col_idx]:.3f})",
                fontsize=11,
            )
            ax.set_xlabel("UMAP 1")
            if col_idx == 0:
                ax.set_ylabel("UMAP 2")

    plt.suptitle(
        "ECG FMs — Balanced UMAP (4 panels per model)   "
        "PTB-XL=Adult, ZZU-pECG=Pediatric   "
        "all panels are 1:1 downsampled before UMAP fit",
        fontsize=14, y=1.001,
    )
    plt.tight_layout()
    fig_path = out_dir / "umap_4panel.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"\n저장: {fig_path}")


if __name__ == "__main__":
    main()
