"""
HEEDB Normal vs Abnormal UMAP with various pretrained encoders
================================================================
Balanced sample from heedb_labels.csv (NORMAL_ECG vs ABNORMAL_ECG),
extract pooled embeddings, visualize via UMAP.

실행:
  python scripts/umap_heedb_founder.py --encoder founder --n_per_class 10000
  python scripts/umap_heedb_founder.py --encoder cpc     --n_per_class 10000
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


HEEDB_ROOT = Path("/home/irteam/ddn-opendata1/h5/heedb/v4.0")
TABLE_CSV = HEEDB_ROOT / "heedb_table.csv"
LABEL_CSV = HEEDB_ROOT / "heedb_labels.csv"

# encoder 이름별 ckpt / 입력 길이 / 샘플링주파수
ENCODER_SPECS = {
    "founder": {
        "ckpt": "/home/irteam/ddn-opendata1/model/ECGFMs/ecg_founder/12_lead_ECGFounder.pth",
        "target_fs": 500,
        "target_length": 2500,   # 5초
    },
    "cpc": {
        "ckpt": "/home/irteam/ddn-opendata1/model/ECGFMs/cpc/last_11597276.ckpt",
        "target_fs": 500,
        "target_length": 5000,   # 10초 (CPC forward가 내부에서 240Hz로 변환)
    },
}


def build_encoder(name: str):
    spec = ENCODER_SPECS[name]
    if name == "founder":
        from src.encoders.ecg_founder import ECGFounderEncoder
        return ECGFounderEncoder(checkpoint=spec["ckpt"]), spec
    if name == "cpc":
        from src.encoders.cpc import CPCEncoder
        return CPCEncoder(checkpoint=spec["ckpt"]), spec
    raise ValueError(f"unknown encoder: {name}")


class HeedbBalancedDataset(Dataset):
    """최소 Dataset — 미리 고른 row DataFrame만 사용.

    row: filepath, fs, sid, is_normal
    반환: signal (12, target_length) at target_fs, is_normal (bool)
    """

    def __init__(self, rows: pd.DataFrame, h5_root: Path,
                 target_fs: int, target_length: int):
        self.rows = rows.reset_index(drop=True)
        self.h5_root = Path(h5_root)
        self.TARGET_FS = target_fs
        self.TARGET_LEN = target_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        fpath = self.h5_root / row["filepath"]
        sid = int(row["sid"])
        fs = int(row["fs"])

        try:
            with h5py.File(fpath, "r") as f:
                sig = f[f"ECG/segments/{sid}/signal"][()].astype(np.float32)
        except Exception:
            sig = np.zeros((12, self.TARGET_LEN), dtype=np.float32)

        if sig.ndim == 2 and sig.shape[0] != 12 and sig.shape[1] == 12:
            sig = sig.T

        # 500Hz로 맞춤
        if fs != self.TARGET_FS:
            new_len = int(round(sig.shape[1] * self.TARGET_FS / fs))
            sig = resample(sig, new_len, axis=1).astype(np.float32)

        # 앞 5초만 사용
        if sig.shape[1] >= self.TARGET_LEN:
            sig = sig[:, :self.TARGET_LEN]
        else:
            pad = np.zeros((sig.shape[0], self.TARGET_LEN - sig.shape[1]),
                           dtype=sig.dtype)
            sig = np.concatenate([sig, pad], axis=1)

        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "signal": torch.from_numpy(sig),
            "is_normal": int(row["is_normal"]),
        }


def build_balanced_rows(n_per_class: int, seed: int = 42) -> pd.DataFrame:
    """NORMAL_ECG vs ABNORMAL_ECG 균형 표본 추출."""
    logging.info(f"라벨 CSV 로드: {LABEL_CSV}")
    labels = pd.read_csv(
        LABEL_CSV,
        usecols=["filepath", "NORMAL_ECG", "ABNORMAL_ECG"],
    )
    logging.info(f"  total rows: {len(labels):,}")

    # mutually exclusive 가정 (앞서 확인). 혹시 겹치면 제외.
    normal = labels[labels["NORMAL_ECG"] & ~labels["ABNORMAL_ECG"]]
    abnormal = labels[labels["ABNORMAL_ECG"] & ~labels["NORMAL_ECG"]]
    logging.info(f"  NORMAL: {len(normal):,} / ABNORMAL: {len(abnormal):,}")

    n = min(n_per_class, len(normal), len(abnormal))
    logging.info(f"  sampling {n:,} per class")

    normal_s = normal.sample(n=n, random_state=seed)[["filepath"]].copy()
    normal_s["is_normal"] = 1
    abnormal_s = abnormal.sample(n=n, random_state=seed)[["filepath"]].copy()
    abnormal_s["is_normal"] = 0

    sampled = pd.concat([normal_s, abnormal_s], ignore_index=True)

    logging.info(f"테이블 CSV 로드: {TABLE_CSV}")
    table = pd.read_csv(TABLE_CSV, usecols=["filepath", "fs", "sid"],
                        low_memory=False)
    merged = sampled.merge(table, on="filepath", how="inner")
    logging.info(f"  merged rows: {len(merged):,} "
                 f"(normal={(merged.is_normal==1).sum():,}, "
                 f"abnormal={(merged.is_normal==0).sum():,})")
    return merged


@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    encoder.eval()
    embs, labels = [], []
    for batch in tqdm(loader, desc="Encoding"):
        x = batch["signal"].to(device, non_blocking=True)
        _, pooled = encoder(x)
        embs.append(pooled.float().cpu().numpy())
        labels.append(batch["is_normal"].numpy())
    return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)


def run_umap_and_plot(emb, labels, out_path: Path, method: str = "umap"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "umap":
        from umap import UMAP
        logging.info(f"UMAP fit_transform on {emb.shape}...")
        reducer = UMAP(n_components=2, random_state=42,
                       n_neighbors=30, min_dist=0.1, metric="cosine")
    else:
        from sklearn.manifold import TSNE
        logging.info(f"t-SNE fit_transform on {emb.shape}...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = reducer.fit_transform(emb)

    from sklearn.metrics import silhouette_score
    sil = silhouette_score(emb, labels, sample_size=min(5000, len(emb)))
    logging.info(f"Silhouette (emb space, Normal vs Abnormal): {sil:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    normal_mask = labels == 1
    ax.scatter(coords[~normal_mask, 0], coords[~normal_mask, 1],
               c="#e74c3c", s=6, alpha=0.45,
               label=f"Abnormal (n={(~normal_mask).sum()})")
    ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1],
               c="#2ecc71", s=6, alpha=0.45,
               label=f"Normal (n={normal_mask.sum()})")
    ax.set_title(
        f"{out_path.stem} — HEEDB Normal vs Abnormal ({method.upper()})\n"
        f"silhouette={sil:.3f}",
        fontsize=13,
    )
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(markerscale=2, fontsize=11)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logging.info(f"saved: {out_path}")

    # 좌표 + 라벨도 같이 저장 (후속 분석용)
    np.savez(
        out_path.with_suffix(".npz"),
        embeddings=emb, labels=labels, coords=coords,
    )
    logging.info(f"saved: {out_path.with_suffix('.npz')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="founder",
                        choices=list(ENCODER_SPECS.keys()))
    parser.add_argument("--n_per_class", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--method", type=str, default="umap",
                        choices=["umap", "tsne"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.output is None:
        args.output = str(
            SCRIPT_DIR
            / f"results/heedb_umap_{args.encoder}_n{args.n_per_class}.png"
        )

    # 1) balanced 샘플링
    rows = build_balanced_rows(args.n_per_class, seed=args.seed)

    # 2) Encoder 로드 + Dataset
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    logging.info(f"device: {device}")
    encoder, spec = build_encoder(args.encoder)
    encoder = encoder.to(device)
    logging.info(
        f"encoder={args.encoder} feature_dim={encoder.feature_dim} "
        f"target_fs={spec['target_fs']} target_length={spec['target_length']}"
    )

    ds = HeedbBalancedDataset(
        rows, h5_root=HEEDB_ROOT,
        target_fs=spec["target_fs"],
        target_length=spec["target_length"],
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4) 임베딩 추출
    emb, labels = extract_embeddings(encoder, loader, device)
    logging.info(f"embeddings: {emb.shape}, labels: {labels.shape}")

    # 5) UMAP + plot
    run_umap_and_plot(emb, labels, Path(args.output), method=args.method)


if __name__ == "__main__":
    main()
