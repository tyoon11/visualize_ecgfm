"""
라벨 재시각화 (기존 임베딩 재사용)
=========================================================
`run_all_embedding_umap.py`로 생성된 results/{tag}/ 디렉토리에서
이미 저장된 임베딩(.npy) + combined UMAP 좌표를 재사용해
다음 두 그림을 추가로 만듭니다:

  1) umap_by_label_each.png      — 라벨 컬럼 각각에 대한 pos/neg
  2) umap_normal_vs_any.png      — Normal(모든 라벨=0) vs Abnormal(하나라도 양성)

사용법:
  python scripts/replot_labels.py --run_dir results/tof_20260423_xxxxxx
"""

import sys
import json
import csv
import argparse
import logging
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def main():
    parser = argparse.ArgumentParser(description="라벨 재시각화 (임베딩 재사용)")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="results/{tag} 경로 (config.json + embeddings/ 포함)")
    parser.add_argument("--abnormal_color", type=str, default="#e74c3c")
    parser.add_argument("--normal_color", type=str, default="#2ecc71")
    parser.add_argument("--point_size", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    run_dir = Path(args.run_dir).resolve()
    emb_dir = run_dir / "embeddings"
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        logging.error(f"config.json 없음: {cfg_path}")
        return

    with open(cfg_path) as f:
        cfg = json.load(f)
    datasets_cfg = cfg["datasets"]
    ds_with_labels = [d for d in datasets_cfg if d.get("label_cols")]
    if not ds_with_labels:
        logging.error("label_cols 가진 데이터셋이 없습니다.")
        return

    # ── 모델 목록 (meta 파일에서 탐지) ──
    meta_files = sorted(emb_dir.glob("*_meta.json"))
    models = []
    for mf in meta_files:
        with open(mf) as f:
            meta = json.load(f)
        models.append({
            "name": meta["model_name"],
            "safe": sanitize(meta["model_name"]),
            "feature_dim": meta["feature_dim"],
        })
    if not models:
        logging.error(f"모델 meta 없음: {emb_dir}")
        return
    logging.info(f"모델 {len(models)}개: {[m['name'] for m in models]}")
    logging.info(f"라벨 데이터셋: "
                 f"{[(d['name'], d['label_cols']) for d in ds_with_labels]}")

    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_models = len(models)
    total_label_cols = sum(len(d["label_cols"]) for d in ds_with_labels)
    n_ds_lbl = len(ds_with_labels)

    fig1, axes1 = plt.subplots(n_models, max(total_label_cols, 1),
                               figsize=(7 * max(total_label_cols, 1), 6 * n_models),
                               squeeze=False)
    fig2, axes2 = plt.subplots(n_models, max(n_ds_lbl, 1),
                               figsize=(7 * max(n_ds_lbl, 1), 6 * n_models),
                               squeeze=False)

    sil_records_each = []
    sil_records_any = []

    for row_idx, m in enumerate(models):
        m_safe = m["safe"]

        # combined UMAP 좌표 로드 (datasets_cfg 순서로 concat된 결과)
        coords_path = emb_dir / f"{m_safe}_umap_coords_combined.npy"
        if not coords_path.exists():
            logging.warning(f"  [{m['name']}] {coords_path.name} 없음, skip")
            continue
        coords = np.load(coords_path)

        # 데이터셋별 슬라이스 계산
        offset = 0
        ds_slices = {}
        for d_cfg in datasets_cfg:
            d_safe = sanitize(d_cfg["name"])
            emb_path = emb_dir / f"{m_safe}_{d_safe}.npy"
            if not emb_path.exists():
                continue
            emb = np.load(emb_path)
            ds_slices[d_cfg["name"]] = (offset, offset + len(emb), emb)
            offset += len(emb)

        col_idx_each = 0
        for ds_idx, d_cfg in enumerate(ds_with_labels):
            dname = d_cfg["name"]
            d_safe = sanitize(dname)
            label_cols = d_cfg["label_cols"]
            lbl_path = emb_dir / f"{m_safe}_{d_safe}_labels.npy"

            if dname not in ds_slices or not lbl_path.exists():
                logging.warning(f"  [{m['name']}/{dname}] 임베딩/라벨 없음, skip")
                col_idx_each += len(label_cols)
                continue

            start, end, emb = ds_slices[dname]
            ds_coords = coords[start:end]
            raw = np.load(lbl_path)
            if raw.ndim == 1:
                raw = raw[:, None]

            # ── Plot 1: 라벨 컬럼 각각 ──
            for li, lc in enumerate(label_cols):
                ax = axes1[row_idx, col_idx_each]
                vals = raw[:, li]
                pos_mask = vals > 0
                neg_mask = ~pos_mask
                n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())

                # 음성 먼저 깔고 양성을 위에 올려 잘 보이게
                ax.scatter(ds_coords[neg_mask, 0], ds_coords[neg_mask, 1],
                           c=args.normal_color, s=args.point_size,
                           alpha=0.45, rasterized=True)
                ax.scatter(ds_coords[pos_mask, 0], ds_coords[pos_mask, 1],
                           c=args.abnormal_color, s=args.point_size,
                           alpha=0.55, rasterized=True)

                binary = pos_mask.astype(int)
                if len(set(binary)) > 1:
                    sil = float(silhouette_score(
                        emb, binary, sample_size=min(10000, len(emb))))
                else:
                    sil = float("nan")
                sil_str = f"{sil:.3f}" if not np.isnan(sil) else "nan"

                ax.set_title(
                    f"{m['name']} — {dname}\n{lc}  (sil={sil_str})",
                    fontsize=10,
                )
                ax.legend(handles=[
                    Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=args.abnormal_color, markersize=8,
                           label=f"{lc}+ (n={n_pos})"),
                    Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=args.normal_color, markersize=8,
                           label=f"{lc}- (n={n_neg})"),
                ], fontsize=8)
                ax.set_xlabel("UMAP 1")
                if col_idx_each == 0:
                    ax.set_ylabel("UMAP 2")

                sil_records_each.append({
                    "model": m["name"], "dataset": dname, "label": lc,
                    "n_pos": n_pos, "n_neg": n_neg, "silhouette": sil,
                })
                col_idx_each += 1

            # ── Plot 2: Normal(전부 0) vs Abnormal(하나라도 양성) ──
            ax = axes2[row_idx, ds_idx]
            any_pos = (raw > 0).any(axis=1)
            any_neg = ~any_pos
            n_pos, n_neg = int(any_pos.sum()), int(any_neg.sum())

            ax.scatter(ds_coords[any_neg, 0], ds_coords[any_neg, 1],
                       c=args.normal_color, s=args.point_size,
                       alpha=0.45, rasterized=True)
            ax.scatter(ds_coords[any_pos, 0], ds_coords[any_pos, 1],
                       c=args.abnormal_color, s=args.point_size,
                       alpha=0.55, rasterized=True)

            binary = any_pos.astype(int)
            if len(set(binary)) > 1:
                sil_any = float(silhouette_score(
                    emb, binary, sample_size=min(10000, len(emb))))
            else:
                sil_any = float("nan")
            sil_str = f"{sil_any:.3f}" if not np.isnan(sil_any) else "nan"

            ax.set_title(
                f"{m['name']} — {dname}\nNormal vs Any-Abnormal  (sil={sil_str})",
                fontsize=10,
            )
            ax.legend(handles=[
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=args.abnormal_color, markersize=8,
                       label=f"Abnormal (any+) (n={n_pos})"),
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=args.normal_color, markersize=8,
                       label=f"Normal (all 0) (n={n_neg})"),
            ], fontsize=8)
            ax.set_xlabel("UMAP 1")
            if ds_idx == 0:
                ax.set_ylabel("UMAP 2")

            sil_records_any.append({
                "model": m["name"], "dataset": dname,
                "labels_used": "|".join(label_cols),
                "n_abnormal": n_pos, "n_normal": n_neg, "silhouette": sil_any,
            })

    plt.figure(fig1.number)
    plt.suptitle("UMAP — per-label pos/neg", fontsize=14, y=1.005)
    plt.tight_layout()
    out1 = run_dir / "umap_by_label_each.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logging.info(f"저장: {out1}")

    plt.figure(fig2.number)
    plt.suptitle("UMAP — Normal vs Any-Abnormal", fontsize=14, y=1.005)
    plt.tight_layout()
    out2 = run_dir / "umap_normal_vs_any.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logging.info(f"저장: {out2}")

    with open(run_dir / "silhouette_per_label.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "dataset", "label", "n_pos", "n_neg", "silhouette"])
        w.writeheader()
        w.writerows(sil_records_each)
    with open(run_dir / "silhouette_normal_vs_any.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "dataset", "labels_used",
            "n_abnormal", "n_normal", "silhouette"])
        w.writeheader()
        w.writerows(sil_records_any)
    logging.info("silhouette CSV 저장 완료")

    logging.info("\n=== silhouette 요약 (Normal vs Any-Abnormal) ===")
    logging.info(f"{'Model':<20} {'Dataset':<12} {'sil':>8} "
                 f"{'#abn':>8} {'#norm':>8}")
    for r in sil_records_any:
        s = f"{r['silhouette']:.4f}" if not np.isnan(r['silhouette']) else "nan"
        logging.info(f"{r['model']:<20} {r['dataset']:<12} {s:>8} "
                     f"{r['n_abnormal']:>8} {r['n_normal']:>8}")


if __name__ == "__main__":
    main()
