"""
ECG Foundation Model 임베딩 추출 + UMAP 시각화 (generic)
=========================================================
임의의 H5 ECG 데이터셋들에 대해 9개 ECG Foundation 모델의 임베딩을 추출하고
UMAP으로 비교 시각화합니다.

사용법:
  python scripts/run_all_embedding_umap.py --config configs/my_datasets.json
  python scripts/run_all_embedding_umap.py --config configs/my_datasets.json --gpus 0,1

출력: results/{timestamp}/
"""

import os
import sys
import json
import argparse
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import H5ECGDataset


# ══════════════════════════════════════════════════════════════════════
# Model Registry — model_dir는 config에서 주입
# ══════════════════════════════════════════════════════════════════════

def build_model_registry(model_dir: Path):
    return [
        {
            "name": "ECG-JEPA",
            "encoder_cls": "src.encoders.ecg_jepa.ECGJEPAEncoder",
            "checkpoint": str(model_dir / "ecg_jepa" / "multiblock_epoch100.pth"),
        },
        {
            "name": "ECG-FM",
            "encoder_cls": "src.encoders.ecg_fm.ECGFMEncoder",
            "checkpoint": str(model_dir / "ecg_fm" / "mimic_iv_ecg_physionet_pretrained.pt"),
        },
        {
            "name": "ECG-Founder",
            "encoder_cls": "src.encoders.ecg_founder.ECGFounderEncoder",
            "checkpoint": str(model_dir / "ecg_founder" / "12_lead_ECGFounder.pth"),
        },
        {
            "name": "ST-MEM",
            "encoder_cls": "src.encoders.st_mem.StMemEncoder",
            "checkpoint": str(model_dir / "st_mem" / "st_mem_vit_base_full.pth"),
        },
        {
            "name": "MERL (ResNet)",
            "encoder_cls": "src.encoders.merl.MerlResNetEncoder",
            "checkpoint": str(model_dir / "merl" / "res18_best_encoder.pth"),
        },
        {
            "name": "MERL (ViT)",
            "encoder_cls": "src.encoders.merl.MerlViTEncoder",
            "checkpoint": str(model_dir / "merl" / "vit_tiny_best_encoder.pth"),
        },
        {
            "name": "ECG-FM-KED",
            "encoder_cls": "src.encoders.ecgfm_ked.EcgFmKEDEncoder",
            "checkpoint": str(
                model_dir / "ecgfm_ked" / "best_valid_all_increase_with_augment_epoch_3.pt"
            ),
        },
        {
            "name": "HuBERT-ECG",
            "encoder_cls": "src.encoders.hubert_ecg.HuBERTECGEncoder",
            "checkpoint": str(model_dir / "hubert_ecg" / "hubert_ecg_base.safetensors"),
        },
        {
            "name": "CPC",
            "encoder_cls": "src.encoders.cpc.CPCEncoder",
            "checkpoint": str(model_dir / "cpc" / "last_11597276.ckpt"),
        },
    ]


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def load_encoder(cfg, device):
    import importlib
    mod_path, cls_name = cfg["encoder_cls"].rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    kwargs = cfg.get("extra_kwargs", {})
    encoder = cls(checkpoint=cfg["checkpoint"], **kwargs)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def extract_embeddings_batched(encoder, dataset, gpu_ids, batch_size=256,
                               num_workers=8, n_samples=0):
    """DataLoader + DataParallel 배치 추론으로 임베딩 추출"""
    if n_samples > 0 and n_samples < len(dataset):
        dataset = Subset(dataset, list(range(n_samples)))

    dp_encoder = nn.DataParallel(encoder, device_ids=gpu_ids) if len(gpu_ids) > 1 else encoder
    dp_encoder.eval()

    total_batch_size = batch_size * len(gpu_ids)
    loader = DataLoader(
        dataset, batch_size=total_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    all_embeddings, all_labels = [], []
    primary_device = torch.device(f"cuda:{gpu_ids[0]}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting", leave=False):
            signals = batch["signal"].to(primary_device)
            labels = batch["label"]
            out = dp_encoder(signals)

            if isinstance(out, tuple):
                pooled = out[1]
            elif isinstance(out, dict):
                pooled = out.get("pooled", out.get("pooled_features"))
            elif out.dim() == 3:
                pooled = out.mean(dim=1)
            else:
                pooled = out

            all_embeddings.append(pooled.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def build_dataset(ds_cfg, default_fs, default_len):
    """config entry → H5ECGDataset"""
    return H5ECGDataset(
        h5_root=ds_cfg["h5_root"],
        table_csv=ds_cfg.get("table_csv"),
        label_csv=ds_cfg.get("label_csv"),
        label_cols=ds_cfg.get("label_cols"),
        target_fs=ds_cfg.get("target_fs", default_fs),
        target_length=ds_cfg.get("target_length", default_len),
        h5_schema=ds_cfg.get("h5_schema", "standard"),
        join_key=ds_cfg.get("join_key", "filepath"),
        filepath_from=ds_cfg.get("filepath_from"),
        filepath_suffix=ds_cfg.get("filepath_suffix", ".h5"),
        expand_8_to_12=ds_cfg.get("expand_8_to_12", False),
    )


def labels_to_class(labels: np.ndarray, label_idx: int = 0,
                    pos_name: str = "Normal", neg_name: str = "Abnormal"):
    """value > 0 → pos_name, else neg_name"""
    return [pos_name if row[label_idx] > 0 else neg_name for row in labels]


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ECG FM 임베딩 + UMAP (generic)")
    parser.add_argument("--config", type=str, required=True,
                        help="데이터셋/모델 config JSON 경로")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="데이터셋당 추출 샘플 수 (0=전체)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=str, default=None,
                        help="GPU IDs (예: '0,1'). 기본: 모든 GPU")
    parser.add_argument("--output_root", type=str, default="results",
                        help="결과 루트 디렉토리 (기본: results/)")
    parser.add_argument("--tag", type=str, default=None,
                        help="타임스탬프 대신 사용할 실행 이름")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ── Config 로드 ──
    with open(args.config) as f:
        cfg = json.load(f)

    model_dir = Path(cfg["model_dir"]).expanduser().resolve()
    datasets_cfg = cfg["datasets"]
    target_fs = cfg.get("target_fs", 500)
    target_length = cfg.get("target_length", 5000)

    # ── GPU 설정 ──
    gpu_ids = [int(g) for g in args.gpus.split(",")] if args.gpus \
              else list(range(torch.cuda.device_count()))
    primary_device = torch.device(f"cuda:{gpu_ids[0]}")
    logging.info(f"GPU {len(gpu_ids)}개 사용: {gpu_ids}")

    # ── 출력 디렉토리 (타임스탬프) ──
    tag = args.tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / tag
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"출력 디렉토리: {out_dir}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ── 데이터셋 로드 ──
    datasets = []
    for ds_cfg in datasets_cfg:
        logging.info(f"Loading dataset: {ds_cfg['name']}")
        ds = build_dataset(ds_cfg, target_fs, target_length)
        n = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))
        logging.info(f"  {ds_cfg['name']}: {n}/{len(ds)} samples, labels={ds.has_labels}")
        datasets.append({"cfg": ds_cfg, "ds": ds, "n": n})

    # ── 모델 필터링 ──
    all_models = build_model_registry(model_dir)
    requested = cfg.get("models")
    if requested:
        models_to_run = [m for m in all_models if m["name"] in requested]
    else:
        models_to_run = all_models
    logging.info(f"실행할 모델: {[m['name'] for m in models_to_run]}")

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: 임베딩 추출
    # ══════════════════════════════════════════════════════════════════
    results = {}
    for m_cfg in models_to_run:
        mname = m_cfg["name"]
        m_safe = sanitize(mname)

        logging.info(f"\n{'='*60}\nModel: {mname}\n{'='*60}")

        # skip_existing 체크
        meta_path = emb_dir / f"{m_safe}_meta.json"
        all_exist = all(
            (emb_dir / f"{m_safe}_{sanitize(d['cfg']['name'])}.npy").exists()
            and (emb_dir / f"{m_safe}_{sanitize(d['cfg']['name'])}_labels.npy").exists()
            for d in datasets
        )

        if args.skip_existing and all_exist and meta_path.exists():
            logging.info(f"  ⏭ 기존 임베딩 사용")
            with open(meta_path) as f:
                meta = json.load(f)
            feature_dim = meta["feature_dim"]
            ds_results = {}
            for d in datasets:
                dname = d["cfg"]["name"]
                d_safe = sanitize(dname)
                ds_results[dname] = {
                    "embeddings": np.load(emb_dir / f"{m_safe}_{d_safe}.npy"),
                    "labels_raw": np.load(emb_dir / f"{m_safe}_{d_safe}_labels.npy"),
                }
            results[mname] = {"feature_dim": feature_dim, "datasets": ds_results}
            continue

        try:
            encoder = load_encoder(m_cfg, primary_device)
            feature_dim = encoder.feature_dim
            logging.info(f"  feature_dim={feature_dim}")

            ds_results = {}
            for d in datasets:
                dname = d["cfg"]["name"]
                d_safe = sanitize(dname)
                logging.info(f"  [{dname}] 임베딩 추출 ({d['n']}개, {len(gpu_ids)} GPUs)...")
                emb, lbl = extract_embeddings_batched(
                    encoder, d["ds"], gpu_ids,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    n_samples=args.n_samples,
                )
                np.save(emb_dir / f"{m_safe}_{d_safe}.npy", emb)
                np.save(emb_dir / f"{m_safe}_{d_safe}_labels.npy", lbl)
                ds_results[dname] = {"embeddings": emb, "labels_raw": lbl}
                logging.info(f"    ✓ {emb.shape}")

            with open(meta_path, "w") as f:
                json.dump({"feature_dim": feature_dim,
                           "model_name": mname,
                           "datasets": [d["cfg"]["name"] for d in datasets]},
                          f, indent=2)

            del encoder
            torch.cuda.empty_cache()
            results[mname] = {"feature_dim": feature_dim, "datasets": ds_results}

        except Exception as e:
            logging.error(f"  ✗ {mname} 실패: {e}")
            import traceback; traceback.print_exc()
            continue

    if not results:
        logging.error("성공한 모델이 없습니다.")
        return

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: UMAP + 시각화
    # ══════════════════════════════════════════════════════════════════
    from umap import UMAP
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    default_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
                      "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    default_markers = ["o", "^", "s", "D", "v", "P", "X", "*"]
    for i, d in enumerate(datasets):
        d["cfg"].setdefault("display_color", default_colors[i % len(default_colors)])
        d["cfg"].setdefault("display_marker", default_markers[i % len(default_markers)])

    n_models = len(results)
    n_datasets = len(datasets)
    has_labels = {d["cfg"]["name"]: d["ds"].has_labels for d in datasets}
    any_labels = any(has_labels.values())
    n_label_cols = sum(1 for v in has_labels.values() if v)

    # Fig 1: 데이터셋별 색상
    fig1, axes1 = plt.subplots(n_models, 1, figsize=(12, 8 * n_models))
    if n_models == 1:
        axes1 = [axes1]

    # Fig 2: Normal/Abnormal per-dataset
    if any_labels:
        fig2, axes2 = plt.subplots(n_models, max(n_label_cols, 1),
                                    figsize=(8 * max(n_label_cols, 1), 7 * n_models),
                                    squeeze=False)

    silhouette_records = []

    for row_idx, (model_name, data) in enumerate(results.items()):
        logging.info(f"\n[UMAP] {model_name}...")
        m_safe = sanitize(model_name)

        all_embs = []
        dataset_labels = []
        for di, d in enumerate(datasets):
            dname = d["cfg"]["name"]
            emb = data["datasets"][dname]["embeddings"]
            all_embs.append(emb)
            dataset_labels.extend([di] * len(emb))
        combined = np.concatenate(all_embs, axis=0)
        dataset_labels = np.array(dataset_labels)

        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(combined)
        np.save(emb_dir / f"{m_safe}_umap_coords_combined.npy", coords)

        sil_ds = float("nan")
        if n_datasets > 1:
            sil_ds = silhouette_score(
                combined, dataset_labels, sample_size=min(10000, len(combined))
            )

        # Plot 1: dataset coloring
        ax = axes1[row_idx]
        offset = 0
        for di, d in enumerate(datasets):
            dname = d["cfg"]["name"]
            n = len(data["datasets"][dname]["embeddings"])
            end = offset + n
            ax.scatter(
                coords[offset:end, 0], coords[offset:end, 1],
                c=d["cfg"]["display_color"],
                marker=d["cfg"]["display_marker"],
                s=6, alpha=0.45, rasterized=True,
                edgecolors="white", linewidths=0.15,
                label=f"{dname} (n={n})",
            )
            offset = end
        title = f"{model_name}  (feature_dim={data['feature_dim']})"
        if n_datasets > 1:
            title += f"  |  dataset-sil={sil_ds:.3f}"
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, markerscale=3, loc="best")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

        # Plot 2: per-dataset Normal/Abnormal
        per_ds_sils = {}
        if any_labels:
            col_idx = 0
            for di, d in enumerate(datasets):
                dname = d["cfg"]["name"]
                if not has_labels[dname]:
                    continue
                ds_offset = sum(
                    len(data["datasets"][datasets[j]["cfg"]["name"]]["embeddings"])
                    for j in range(di)
                )
                ds_emb = data["datasets"][dname]["embeddings"]
                raw = data["datasets"][dname]["labels_raw"]
                label_col = d["cfg"].get("label_col")
                if label_col and label_col in d["ds"].label_cols:
                    label_idx = d["ds"].label_cols.index(label_col)
                else:
                    label_idx = 0

                # 라벨 의미 설정 (config에서 override 가능)
                # 기본: PTB-XL 스타일 — label=1이 Normal(녹색), label=0이 Abnormal(빨강)
                pos_name = d["cfg"].get("positive_label", "Normal")
                neg_name = d["cfg"].get("negative_label", "Abnormal")
                pos_color = d["cfg"].get("positive_color", "#2ecc71")
                neg_color = d["cfg"].get("negative_color", "#e74c3c")

                class_str = labels_to_class(raw, label_idx, pos_name, neg_name)
                ds_coords = coords[ds_offset:ds_offset + len(ds_emb)]

                binary = [1 if s == pos_name else 0 for s in class_str]
                sil_na = (silhouette_score(ds_emb, binary,
                                           sample_size=min(10000, len(ds_emb)))
                          if len(set(binary)) > 1 else float("nan"))
                per_ds_sils[dname] = sil_na

                ax = axes2[row_idx, col_idx]
                colors = [pos_color if s == pos_name else neg_color for s in class_str]
                ax.scatter(ds_coords[:, 0], ds_coords[:, 1], c=colors, s=5,
                           alpha=0.45, rasterized=True)
                n_pos = binary.count(1)
                n_neg = binary.count(0)
                legend_na = [
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=pos_color,
                           markersize=8, label=f"{pos_name} (n={n_pos})"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=neg_color,
                           markersize=8, label=f"{neg_name} (n={n_neg})"),
                ]
                ax.legend(handles=legend_na, fontsize=8)
                sil_str = f"{sil_na:.3f}" if not np.isnan(sil_na) else "nan"
                ax.set_title(
                    f"{model_name} — {dname}\n{pos_name} vs {neg_name} ({label_col}, sil={sil_str})",
                    fontsize=11,
                )
                ax.set_xlabel("UMAP 1")
                if col_idx == 0:
                    ax.set_ylabel("UMAP 2")
                col_idx += 1

        rec = {"model": model_name, "feature_dim": data["feature_dim"],
               "dataset_separation": sil_ds}
        for dname, sil in per_ds_sils.items():
            rec[f"{dname}_pos_vs_neg"] = sil
        silhouette_records.append(rec)

    # 저장
    plt.figure(fig1.number)
    plt.suptitle(
        f"ECG FMs — UMAP by Dataset ({' / '.join(d['cfg']['name'] for d in datasets)})",
        fontsize=14, y=1.005,
    )
    plt.tight_layout()
    fig1.savefig(out_dir / "umap_by_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logging.info(f"그림 저장: {out_dir}/umap_by_dataset.png")

    if any_labels:
        plt.figure(fig2.number)
        plt.suptitle("ECG FMs — binary label per dataset",
                     fontsize=14, y=1.005)
        plt.tight_layout()
        fig2.savefig(out_dir / "umap_by_label.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logging.info(f"그림 저장: {out_dir}/umap_by_label.png")

    # Silhouette CSV
    import csv
    csv_path = out_dir / "silhouette_scores.csv"
    all_keys = set()
    for r in silhouette_records:
        all_keys.update(r.keys())
    fieldnames = ["model", "feature_dim", "dataset_separation"] + \
                 [k for k in sorted(all_keys) if k.endswith("_pos_vs_neg")]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(silhouette_records)
    logging.info(f"실루엣 저장: {csv_path}")

    # 터미널 요약
    logging.info("\n" + "=" * 70)
    logging.info(f"{'Model':<20} {'Dim':>5} {'DatasetSil':>11}")
    logging.info("-" * 70)
    for rec in silhouette_records:
        ds_s = rec["dataset_separation"]
        ds_str = f"{ds_s:>11.4f}" if not np.isnan(ds_s) else f"{'nan':>11}"
        logging.info(f"{rec['model']:<20} {rec['feature_dim']:>5} {ds_str}")
    logging.info("=" * 70)
    logging.info(f"\n✅ 완료. 결과: {out_dir}")


if __name__ == "__main__":
    main()
