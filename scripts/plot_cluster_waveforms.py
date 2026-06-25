"""
클러스터 대표 파형 (medoid) 플롯
================================
cluster_heedb.py 결과(KMeans 라벨 + 임베딩 캐시)에서 각 클러스터의
medoid(L2정규화 임베딩이 클러스터 중심에 cosine 최댓값인 실제 ECG)를 찾아
원본 12-lead 파형을 그린다.

  - 클러스터별 12-lead 그리드 (top-1 medoid)
  - 전 클러스터 Lead II 비교 스트립 (medoid)

실행:
  CUDA_VISIBLE_DEVICES="" python scripts/plot_cluster_waveforms.py \
      --cluster_ts 20260619_160046 --n_samples 100000 --seed 42
"""
import os, sys, argparse, logging
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from scripts.umap_heedb_founder import HEEDB_ROOT  # noqa: E402
from scripts.cluster_heedb import build_random_rows, INTEREST_LABELS  # noqa: E402

# HEEDB 고정 채널 순서 (v4.0 Readme.md) — 표준 순서 아님!
LEAD_NAMES = ["I", "II", "III", "V1", "V2", "V3",
              "V4", "V5", "V6", "aVF", "aVL", "aVR"]
LEAD_II = 1  # Lead II는 index 1


def load_signal(filepath, sid, fs, seconds=5.0):
    """원본 12-lead 반환 (12, n), n = seconds*fs. sig_name으로 순서 검증."""
    fp = Path(HEEDB_ROOT) / filepath
    with h5py.File(fp, "r") as f:
        seg = f[f"ECG/segments/{sid}"]
        sig = seg["signal"][()].astype(np.float32)
        names = None
        for key in ("sig_name", "channel_name"):
            if key in seg:
                names = [n.decode() if isinstance(n, bytes) else str(n)
                         for n in seg[key][()]]
                break
            if key in f["ECG"]:
                names = [n.decode() if isinstance(n, bytes) else str(n)
                         for n in f["ECG"][key][()]]
                break
    if sig.ndim == 2 and sig.shape[0] != 12 and sig.shape[1] == 12:
        sig = sig.T
    sig = np.nan_to_num(sig)
    n = int(seconds * fs)
    return sig[:, :n], fs, names


def dominant_labels(rates_row, k=3):
    s = rates_row.drop(labels=["size"], errors="ignore").sort_values(
        ascending=False)
    return ", ".join(f"{n} {v:.0f}%" for n, v in s.head(k).items() if v > 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_ts", required=True)
    p.add_argument("--encoder", default="founder")
    p.add_argument("--n_samples", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_medoid", type=int, default=3,
                   help="클러스터당 medoid 후보 (lead II 비교에 겹쳐 그림)")
    p.add_argument("--pool", type=int, default=60,
                   help="중심 근처 후보 풀 크기 (저전압 리드 걸러낼 범위)")
    p.add_argument("--ptp_min", type=float, default=0.1,
                   help="모든 리드가 넘어야 할 최소 peak-to-peak (mV)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    res_dir = SCRIPT_DIR / "results" / args.cluster_ts
    emb_cache = (SCRIPT_DIR / "results" / "embeddings" /
                 f"{args.encoder}_heedb_cluster_n{args.n_samples}_s{args.seed}.npz")
    z = np.load(emb_cache, allow_pickle=True)
    emb = z["emb"].astype(np.float32)
    cache_is_normal = z["row_is_normal"]

    cr = np.load(res_dir / f"cluster_result_{args.encoder}.npz")
    km = cr["kmeans"]
    assert len(km) == len(emb), "라벨/임베딩 길이 불일치"

    # 결정적 재생성으로 filepath/fs/sid 복원 + 정렬 검증
    rows = build_random_rows(args.n_samples, seed=args.seed).reset_index(drop=True)
    assert np.array_equal(rows["is_normal"].to_numpy(), cache_is_normal), \
        "재생성 rows 정렬이 캐시와 불일치 — 샘플링 비결정적?"
    logging.info("정렬 검증 통과 (rows ↔ 임베딩/라벨)")

    # 클러스터별 라벨 비율 (제목용)
    df = rows.copy(); df["cluster"] = km
    rates = df.groupby("cluster")[INTEREST_LABELS].mean() * 100.0
    rates["size"] = df.groupby("cluster").size()

    # L2정규화 → 클러스터 중심 → cosine 내림차순 후보 풀
    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    clusters = sorted(int(c) for c in set(km) if c != -1)
    pool = {}
    for c in clusters:
        idx = np.where(km == c)[0]
        cen = embn[idx].mean(0); cen /= (np.linalg.norm(cen) + 1e-8)
        sim = embn[idx] @ cen
        pool[c] = idx[np.argsort(-sim)][:args.pool]  # 중심에 가까운 순

    def is_clean(filepath, sid, fs, ptp_min):
        """12리드 전부 충분한 진폭(저전압/zero-lead 없음)인지 검사."""
        sig, _, _ = load_signal(filepath, sid, fs)
        return bool(np.all(np.ptp(sig, axis=1) > ptp_min))

    # 클러스터별로 cosine 순서 유지하며 깨끗한(12리드 정상) 후보 선별
    medoids = {}
    for c in clusters:
        clean = []
        for gi in pool[c]:
            r = rows.iloc[int(gi)]
            if is_clean(r["filepath"], int(r["sid"]), int(r["fs"]), args.ptp_min):
                clean.append(int(gi))
            if len(clean) >= args.n_medoid:
                break
        if not clean:  # 전부 zero-lead면 그냥 최상위 사용
            clean = [int(pool[c][0])]
        medoids[c] = np.array(clean)
        logging.info(f"cluster {c}: medoid={medoids[c][0]} "
                     f"(clean {len(clean)}/{len(pool[c])} 검사)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 1) 클러스터별 12-lead (top-1 medoid) ----
    for c in clusters:
        gi = int(medoids[c][0])
        r = rows.iloc[gi]
        sig, fs, names = load_signal(r["filepath"], int(r["sid"]), int(r["fs"]))
        names = names if names and len(names) == 12 else LEAD_NAMES
        t = np.arange(sig.shape[1]) / fs
        fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=True)
        for li, ax in enumerate(axes.flat):
            ax.plot(t, sig[li], lw=0.7, color="#222")
            ax.set_title(names[li], fontsize=9, loc="left")
            ax.grid(alpha=0.25)
            ax.margins(x=0)
        fig.suptitle(
            f"Cluster {c} medoid  (n={int(rates.loc[c,'size'])})  —  "
            f"{dominant_labels(rates.loc[c])}\n{r['filepath']} sid={int(r['sid'])}",
            fontsize=12)
        fig.supxlabel("time (s)"); fig.supylabel("mV")
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        out = res_dir / f"waveform_cluster{c}_{args.encoder}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        logging.info(f"saved: {out}")

    # ---- 2) 전 클러스터 Lead II 비교 (medoid n_medoid개 겹침) ----
    fig, axes = plt.subplots(len(clusters), 1,
                             figsize=(14, 2.1 * len(clusters)), sharex=True)
    if len(clusters) == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors
    for ax, c in zip(axes, clusters):
        for j, gi in enumerate(medoids[c]):
            r = rows.iloc[int(gi)]
            sig, fs, _ = load_signal(r["filepath"], int(r["sid"]), int(r["fs"]))
            t = np.arange(sig.shape[1]) / fs
            ax.plot(t, sig[LEAD_II], lw=0.6, alpha=0.8,
                    color=colors[j % 10])
        ax.set_ylabel(f"c{c}\n(n={int(rates.loc[c,'size'])})",
                      rotation=0, ha="right", va="center", fontsize=9)
        ax.set_title(dominant_labels(rates.loc[c]), fontsize=9, loc="left")
        ax.grid(alpha=0.25); ax.margins(x=0)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Lead II medoids by cluster ({args.encoder}, "
                 f"KMeans k={len(clusters)})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = res_dir / f"waveform_leadII_compare_{args.encoder}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    logging.info(f"saved: {out}")
    logging.info("done.")


if __name__ == "__main__":
    main()
