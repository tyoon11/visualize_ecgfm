"""
HEEDB Unsupervised Clustering
=============================
전체 HEEDB에서 라벨 무관 랜덤 표본을 뽑아 사전학습 인코더 임베딩을 추출하고,
KMeans(스윕) + HDBSCAN 비지도 클러스터링 후 UMAP(클러스터 색칠)로 시각화한다.
추가로 발견된 클러스터를 주요 진단 라벨과 교차표로 비교해 해석을 돕는다.

실행 예:
  CUDA_VISIBLE_DEVICES=6 python scripts/cluster_heedb.py \
      --encoder founder --n_samples 100000

캐시:
  results/embeddings/{enc}_heedb_cluster_n{N}_s{seed}.npz  (emb, coords, labels, rows)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 기존 founder 스크립트의 Dataset/encoder/extract 재사용
from scripts.umap_heedb_founder import (  # noqa: E402
    HEEDB_ROOT,
    TABLE_CSV,
    LABEL_CSV,
    HeedbBalancedDataset,
    build_encoder,
    extract_embeddings,
)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# 해석용으로 같이 들고 다닐 대표 라벨 컬럼
INTEREST_LABELS = [
    "NORMAL_ECG",
    "ABNORMAL_ECG",
    "ATRIAL_FIBRILLATION",
    "SINUS_BRADYCARDIA",
    "SINUS_TACHYCARDIA",
    "RIGHT_BUNDLE_BRANCH_BLOCK",
    "LEFT_BUNDLE_BRANCH_BLOCK",
    "PREMATURE_VENTRICULAR_COMPLEXES",
    "LEFT_VENTRICULAR_HYPERTROPHY",
    "VENTRICULAR_PACED_RHYTHM",
    "PEDIATRIC_ECG_ANALYSIS",
]


def build_random_rows(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """전체 HEEDB에서 라벨 무관 랜덤 표본. 해석용 라벨 + is_normal 동반."""
    logging.info(f"라벨 CSV 로드: {LABEL_CSV}")
    usecols = ["filepath"] + INTEREST_LABELS
    labels = pd.read_csv(LABEL_CSV, usecols=usecols)
    logging.info(f"  total rows: {len(labels):,}")

    n = min(n_samples, len(labels))
    sampled = labels.sample(n=n, random_state=seed).reset_index(drop=True)
    sampled["is_normal"] = (
        sampled["NORMAL_ECG"] & ~sampled["ABNORMAL_ECG"]
    ).astype(int)
    logging.info(f"  sampled {n:,} (normal={sampled['is_normal'].sum():,})")

    logging.info(f"테이블 CSV 로드: {TABLE_CSV}")
    table = pd.read_csv(
        TABLE_CSV, usecols=["filepath", "fs", "sid"], low_memory=False
    )
    merged = sampled.merge(table, on="filepath", how="inner")
    logging.info(f"  merged rows: {len(merged):,}")
    return merged


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def run_kmeans_sweep(emb: np.ndarray, k_list, seed: int):
    """MiniBatchKMeans 스윕. silhouette(샘플) 최고 k 선택."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.RandomState(seed)
    sil_idx = rng.choice(
        len(emb), size=min(10000, len(emb)), replace=False
    )

    results = []
    for k in k_list:
        km = MiniBatchKMeans(
            n_clusters=k, random_state=seed, batch_size=4096,
            n_init=5, max_iter=300,
        )
        lab = km.fit_predict(emb)
        sil = silhouette_score(emb[sil_idx], lab[sil_idx], metric="euclidean")
        logging.info(f"  KMeans k={k:>3d}  silhouette={sil:.4f}")
        results.append((k, sil, lab, km.inertia_))
    best = max(results, key=lambda r: r[1])
    logging.info(f"  -> best k={best[0]} (silhouette={best[1]:.4f})")
    return best, results


def run_hdbscan(emb_2d: np.ndarray, min_cluster_size: int):
    """UMAP 2D 좌표 위에서 HDBSCAN (밀도 기반, k 불필요)."""
    from sklearn.cluster import HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=10,
    )
    lab = clusterer.fit_predict(emb_2d)
    n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
    n_noise = int((lab == -1).sum())
    logging.info(
        f"  HDBSCAN: {n_clusters} clusters, "
        f"noise={n_noise} ({100*n_noise/len(lab):.1f}%)"
    )
    return lab


def plot_clusters(coords, cluster_lab, is_normal, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # (a) 클러스터 색칠
    ax = axes[0]
    uniq = sorted(set(cluster_lab))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    for i, c in enumerate(uniq):
        m = cluster_lab == c
        color = "#bbbbbb" if c == -1 else cmap(i)
        lbl = "noise" if c == -1 else f"c{c} (n={m.sum()})"
        ax.scatter(coords[m, 0], coords[m, 1], s=5, alpha=0.45,
                   color=color, label=lbl)
    ax.set_title(f"{title}\nclusters={len([c for c in uniq if c!=-1])}")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    if len(uniq) <= 25:
        ax.legend(markerscale=2, fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.2)

    # (b) Normal/Abnormal 참고 색칠
    ax = axes[1]
    nm = is_normal == 1
    ax.scatter(coords[~nm, 0], coords[~nm, 1], s=5, alpha=0.4,
               c="#e74c3c", label=f"Abnormal (n={(~nm).sum()})")
    ax.scatter(coords[nm, 0], coords[nm, 1], s=5, alpha=0.4,
               c="#2ecc71", label=f"Normal (n={nm.sum()})")
    ax.set_title("reference: Normal vs Abnormal")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=2, fontsize=10)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logging.info(f"saved: {out_path}")
    plt.close(fig)


def cluster_label_crosstab(cluster_lab, rows: pd.DataFrame, out_csv: Path):
    """클러스터별 주요 라벨 비율(%) 교차표."""
    df = rows.copy()
    df["cluster"] = cluster_lab
    df = df[df["cluster"] != -1]
    rates = df.groupby("cluster")[INTEREST_LABELS].mean() * 100.0
    rates["size"] = df.groupby("cluster").size()
    rates = rates.round(1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(out_csv)
    logging.info(f"saved crosstab: {out_csv}")
    with pd.option_context("display.width", 200,
                           "display.max_columns", 50):
        logging.info("클러스터별 라벨 비율(%):\n" + rates.to_string())
    return rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", type=str, default="founder",
                   choices=["founder", "cpc"])
    p.add_argument("--n_samples", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k_list", type=int, nargs="+",
                   default=[5, 8, 10, 15, 20, 30])
    p.add_argument("--hdbscan_min_size", type=int, default=500)
    p.add_argument("--timestamp", type=str, required=True,
                   help="결과 폴더 results/<timestamp>/")
    p.add_argument("--recompute", action="store_true",
                   help="임베딩 캐시 무시하고 재추출")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out_dir = SCRIPT_DIR / "results" / args.timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = SCRIPT_DIR / "results" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    cache = emb_dir / (
        f"{args.encoder}_heedb_cluster_n{args.n_samples}_s{args.seed}.npz"
    )

    # 1) 임베딩 (캐시 우선)
    if cache.exists() and not args.recompute:
        logging.info(f"임베딩 캐시 로드: {cache}")
        z = np.load(cache, allow_pickle=True)
        emb = z["emb"]
        rows = pd.DataFrame({c: z[f"row_{c}"] for c in
                             (["is_normal"] + INTEREST_LABELS)})
    else:
        rows = build_random_rows(args.n_samples, seed=args.seed)
        device = torch.device(
            args.device if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")
        encoder, spec = build_encoder(args.encoder)
        encoder = encoder.to(device)
        logging.info(f"encoder={args.encoder} dim={encoder.feature_dim} "
                     f"fs={spec['target_fs']} len={spec['target_length']}")
        ds = HeedbBalancedDataset(
            rows, h5_root=HEEDB_ROOT,
            target_fs=spec["target_fs"],
            target_length=spec["target_length"],
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        emb, _ = extract_embeddings(encoder, loader, device)
        logging.info(f"embeddings: {emb.shape}")
        save = {"emb": emb}
        for c in (["is_normal"] + INTEREST_LABELS):
            save[f"row_{c}"] = rows[c].to_numpy()
        np.savez_compressed(cache, **save)
        logging.info(f"캐시 저장: {cache}")

    is_normal = rows["is_normal"].to_numpy().astype(int)

    # 2) L2 정규화 (cosine 기하 맞춤)
    embn = l2norm(emb.astype(np.float32))

    # 3) KMeans 스윕
    logging.info("=== KMeans 스윕 ===")
    (best_k, best_sil, best_lab, _), _ = run_kmeans_sweep(
        embn, args.k_list, args.seed)

    # 4) UMAP 2D
    from umap import UMAP
    logging.info(f"UMAP fit_transform on {embn.shape} ...")
    coords = UMAP(n_components=2, random_state=42, n_neighbors=30,
                  min_dist=0.1, metric="cosine").fit_transform(embn)

    # 5) HDBSCAN (2D 위)
    logging.info("=== HDBSCAN ===")
    hdb_lab = run_hdbscan(coords, args.hdbscan_min_size)

    # 6) plot + crosstab
    plot_clusters(
        coords, best_lab, is_normal,
        title=f"HEEDB KMeans k={best_k} ({args.encoder}, n={len(emb)})",
        out_path=out_dir / f"cluster_kmeans_k{best_k}_{args.encoder}.png",
    )
    cluster_label_crosstab(
        best_lab, rows,
        out_dir / f"crosstab_kmeans_k{best_k}_{args.encoder}.csv")

    if hdb_lab is not None:
        plot_clusters(
            coords, hdb_lab, is_normal,
            title=f"HEEDB HDBSCAN ({args.encoder}, n={len(emb)})",
            out_path=out_dir / f"cluster_hdbscan_{args.encoder}.png",
        )
        cluster_label_crosstab(
            hdb_lab, rows,
            out_dir / f"crosstab_hdbscan_{args.encoder}.csv")

    # 좌표/라벨 저장 (후속 분석용)
    np.savez_compressed(
        out_dir / f"cluster_result_{args.encoder}.npz",
        coords=coords, kmeans=best_lab,
        hdbscan=(hdb_lab if hdb_lab is not None else np.array([])),
        is_normal=is_normal, best_k=best_k,
    )
    logging.info("done.")


if __name__ == "__main__":
    main()
