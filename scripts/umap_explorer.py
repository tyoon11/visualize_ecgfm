"""
ECG Foundation Model — Interactive UMAP Explorer
=================================================
사전 추출된 임베딩(.npy) + 미리 계산된 UMAP 좌표 캐시를 읽어
여러 프리셋을 탭으로 비교합니다.

실행:
  streamlit run umap_explorer.py -- --result_dir results
  streamlit run umap_explorer.py -- --result_dir results/20260423_181740

`result_dir` 아래에 `embeddings/`가 있거나, 또는 `result_dir` 자체가
임베딩 디렉토리여도 됩니다.

캐시 사전 생성:
  python scripts/precompute_umap_cache.py --emb_dir results/embeddings
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from umap import UMAP

# ─── CLI args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, default="results",
                    help="임베딩 디렉토리 또는 그 부모 (embeddings/ 포함)")
try:
    args, _ = parser.parse_known_args()
    RESULT_DIR = Path(args.result_dir).resolve()
except SystemExit:
    RESULT_DIR = Path("results").resolve()

if (RESULT_DIR / "embeddings").is_dir():
    EMB_DIR = RESULT_DIR / "embeddings"
elif RESULT_DIR.is_dir() and any(RESULT_DIR.glob("*.npy")):
    EMB_DIR = RESULT_DIR
else:
    EMB_DIR = RESULT_DIR / "embeddings"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = PROJECT_ROOT / "labels"

# ─── 팔레트 / 모델 힌트 / 라벨 CSV 매핑 ─────────────────────────────────────
PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44",
           "#66CCEE", "#AA3377", "#BBBBBB", "#332288"]
MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]

MODEL_ORDER_HINT = [
    "CPC", "ECG-FM", "ECG-FM-KED", "ECG-Founder", "ECG-JEPA",
    "HuBERT-ECG", "MERL (ResNet)", "MERL (ViT)", "ST-MEM",
]

LABEL_CSV_HINTS = {
    "ptbxl":   "ptbxl_super_bench_labels.csv",
    "zzu":     "zzu_bench_labels.csv",
    "PTB-XL":  "ptbxl_super_bench_labels.csv",
    "ZZU-pECG": "zzu_bench_labels.csv",
}

# 데이터셋명 → (table_csv, age_col, age_scale) 휴리스틱 (config.json 없을 때)
AGE_TABLE_HINTS: dict[str, tuple[str, str, float]] = {
    "ptbxl":    ("/home/irteam/ddn-opendata1/h5/physionet/v2.0/ptbxl_table.csv", "age", 100.0),
    "PTB-XL":   ("/home/irteam/ddn-opendata1/h5/physionet/v2.0/ptbxl_table.csv", "age", 100.0),
    "zzu":      ("/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0/ecg_table.csv",     "age", 100.0),
    "ZZU-pECG": ("/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0/ecg_table.csv",     "age", 100.0),
}

# ─── 프리셋 정의 ────────────────────────────────────────────────────────────
# tag/n_neighbors/min_dist/metric/do_l2 만 정의. 캐시 파일명은 cache_filenames_for() 가 동적으로 생성.
PRESETS: dict[str, dict] = {
    "원본 재현 (euclidean, L2 OFF)": {
        "tag": "orig",
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
        "do_l2": False,
        "desc": "euclidean + L2 OFF (run_all_embedding_umap.py와 동일)",
    },
    "euclidean + L2 ON": {
        "tag": "euclideanL2",
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
        "do_l2": True,
        "desc": "euclidean + L2 ON (원본 재현과 L2만 토글한 ablation 짝)",
    },
    "cosine + L2 ON": {
        "tag": "cosineL2",
        "n_neighbors": 30,
        "min_dist": 0.05,
        "metric": "cosine",
        "do_l2": True,
        "desc": "cosine + L2 (foundation model 비교용)",
    },
}


def cache_filenames_for(safe: str, preset_tag: str, ds_tuple: tuple) -> tuple[tuple, str]:
    """
    데이터셋 조합에 따라 캐시 파일명을 생성.
    역호환: ds_tuple == ('ptbxl', 'zzu') 인 경우만 기존 무접미사 파일명 사용.
    """
    is_legacy = tuple(ds_tuple) == ("ptbxl", "zzu")
    suffix = "" if is_legacy else "__" + "_".join(ds_tuple)
    if preset_tag == "orig":
        save_fn = f"{safe}_umap_coords{suffix}.npy"
        load_candidates = [save_fn]
        if is_legacy:
            load_candidates.append(f"{safe}_umap_coords_all.npy")
        return tuple(load_candidates), save_fn
    if preset_tag == "cosineL2":
        save_fn = f"{safe}_umap_coords_cosineL2{suffix}.npy"
        return (save_fn,), save_fn
    if preset_tag == "euclideanL2":
        save_fn = f"{safe}_umap_coords_euclideanL2{suffix}.npy"
        return (save_fn,), save_fn
    # user / unknown — 캐시 사용 안 함
    return (), ""


# ─── 유틸 ────────────────────────────────────────────────────────────────────
def sanitize(name: str) -> str:
    """run_all_embedding_umap.py / plot_normal_abnormal.py 와 동일한 규칙."""
    return (name.replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace("/", "_"))


def discover_models_and_datasets(emb_dir: Path):
    models: dict[str, dict] = {}
    datasets: set[str] = set()

    for f in emb_dir.glob("*_meta.json"):
        try:
            with open(f) as fp:
                meta = json.load(fp)
        except Exception:
            continue
        name = meta.get("model_name") or f.stem.replace("_meta", "")
        safe = meta.get("safe") or sanitize(name)
        models[name] = {"safe": safe, "feature_dim": meta.get("feature_dim")}
        for ds in meta.get("datasets", []) or []:
            datasets.add(ds)

    for f in emb_dir.glob("*_meta.npz"):
        safe = f.stem.replace("_meta", "")
        display = next(
            (m for m in MODEL_ORDER_HINT if sanitize(m) == safe),
            safe,
        )
        if display in models:
            continue
        try:
            fd = int(np.load(f, allow_pickle=True)["feature_dim"])
        except Exception:
            fd = None
        models[display] = {"safe": safe, "feature_dim": fd}

    known_safes = {m["safe"] for m in models.values()}
    for npy in emb_dir.glob("*.npy"):
        stem = npy.stem
        if stem.endswith("_labels") or "_umap_" in stem or stem.endswith("_umap_coords"):
            continue
        matched_safe = None
        for safe in sorted(known_safes, key=len, reverse=True):
            if stem.startswith(safe + "_"):
                matched_safe = safe
                break
        if matched_safe is None:
            for hint in sorted(MODEL_ORDER_HINT, key=lambda x: len(sanitize(x)), reverse=True):
                s = sanitize(hint)
                if stem.startswith(s + "_"):
                    matched_safe = s
                    if hint not in models:
                        models[hint] = {"safe": s, "feature_dim": None}
                    known_safes.add(s)
                    break
        if matched_safe is None:
            continue
        ds_name = stem[len(matched_safe) + 1:]
        datasets.add(ds_name)

    model_names = sorted(models.keys(),
                         key=lambda m: (MODEL_ORDER_HINT.index(m)
                                        if m in MODEL_ORDER_HINT else 999, m))
    return model_names, sorted(datasets), models


def find_label_cols(ds_name: str, n_cols: int, config_info: dict | None):
    if config_info:
        lc = config_info.get("label_cols")
        if lc:
            return list(lc)[:n_cols] + [f"label_{i}" for i in range(len(lc), n_cols)]
        single = config_info.get("label_col")
        if single and n_cols == 1:
            return [single]
    csv_name = LABEL_CSV_HINTS.get(ds_name)
    if csv_name is None:
        for k, v in LABEL_CSV_HINTS.items():
            if k.lower() == ds_name.lower():
                csv_name = v
                break
    if csv_name:
        csv_path = LABELS_DIR / csv_name
        if csv_path.exists():
            try:
                cols = list(pd.read_csv(csv_path, nrows=0).columns)
                skip = {"filepath", "path", "file", "dataset",
                        "pid", "rid", "oid", "age", "sex"}
                label_cols = [c for c in cols if c not in skip]
                if len(label_cols) >= n_cols:
                    return label_cols[:n_cols]
                if label_cols:
                    return label_cols + [f"label_{i}" for i in range(len(label_cols), n_cols)]
            except Exception:
                pass
    return [f"label_{i}" for i in range(n_cols)]


@st.cache_data(show_spinner=False)
def load_embeddings(emb_dir_str: str, model_safe: str, dataset_name: str):
    emb_dir = Path(emb_dir_str)
    emb_path = emb_dir / f"{model_safe}_{dataset_name}.npy"
    lbl_path = emb_dir / f"{model_safe}_{dataset_name}_labels.npy"
    if not emb_path.exists():
        return None, None
    emb = np.load(emb_path)
    lbl = np.load(lbl_path) if lbl_path.exists() else None
    return emb, lbl


@st.cache_data(show_spinner=False)
def compute_umap_cached(emb_bytes: bytes, shape: tuple, n_neighbors: int,
                        min_dist: float, metric: str, seed: int):
    arr = np.frombuffer(emb_bytes, dtype=np.float32).reshape(shape)
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                   min_dist=min_dist, metric=metric, random_state=seed)
    return reducer.fit_transform(arr)


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return emb / norms


def subsample(emb, labels, n):
    if n >= len(emb):
        return emb, labels
    rng = np.random.RandomState(42)
    idx = rng.choice(len(emb), n, replace=False)
    return emb[idx], (labels[idx] if labels is not None else None)


def compute_metrics(emb, binary):
    if len(set(binary.tolist())) < 2:
        return float("nan"), float("nan")
    n = min(5000, len(emb))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(emb), n, replace=False)
    try:
        sil = silhouette_score(emb[idx], binary[idx])
    except Exception:
        sil = float("nan")
    try:
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
        bacc = cross_val_score(knn, emb[idx], binary[idx],
                               cv=5, scoring="balanced_accuracy").mean()
    except Exception:
        bacc = float("nan")
    return (round(sil, 4) if np.isfinite(sil) else float("nan"),
            round(bacc, 4) if np.isfinite(bacc) else float("nan"))


# ─── 캐시 좌표 로드/저장 ─────────────────────────────────────────────────────
def try_load_cached_combined(emb_dir: Path, ordered_ds: list[str],
                             sizes: dict[str, int],
                             load_candidates: tuple[str, ...]):
    expected_n = sum(sizes[d] for d in ordered_ds)
    for fn in load_candidates:
        p = emb_dir / fn
        if not p.exists():
            continue
        try:
            coords = np.load(p)
        except Exception:
            continue
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != expected_n:
            continue
        out, off = {}, 0
        for d in ordered_ds:
            n = sizes[d]
            out[d] = coords[off:off + n]
            off += n
        return out, p.name
    return None, None


def save_cached_combined(emb_dir: Path, save_filename: str,
                         ordered_ds: list[str], coords_by_ds: dict):
    if not save_filename:
        return
    parts = [coords_by_ds[d] for d in ordered_ds]
    coords = np.concatenate(parts, axis=0)
    np.save(emb_dir / save_filename, coords)


def compute_combined_for_model(emb_dir: Path, safe: str, ordered_ds: list[str],
                               emb_by_ds: dict, preset: dict, seed: int,
                               try_cache: bool, save_after_compute: bool):
    """한 모델에 대해 (preset, ordered_ds) 조합의 combined-fit 좌표를 얻는다."""
    load_candidates, save_fn = cache_filenames_for(
        safe, preset["tag"], tuple(ordered_ds))
    sizes = {d: len(emb_by_ds[d]) for d in ordered_ds}

    if try_cache and load_candidates:
        cached, hit = try_load_cached_combined(
            emb_dir, ordered_ds, sizes, load_candidates)
        if cached is not None:
            return cached, hit

    parts = []
    for d in ordered_ds:
        e = emb_by_ds[d].astype(np.float32)
        if preset["do_l2"]:
            e = l2_normalize(e)
        parts.append(e)
    concat = np.concatenate(parts, axis=0)
    coords = compute_umap_cached(
        concat.tobytes(), concat.shape,
        preset["n_neighbors"], float(preset["min_dist"]),
        preset["metric"], int(seed),
    )
    out, off = {}, 0
    for d in ordered_ds:
        n = sizes[d]
        out[d] = coords[off:off + n]
        off += n

    if save_after_compute and save_fn:
        try:
            save_cached_combined(emb_dir, save_fn, ordered_ds, out)
        except Exception:
            pass
    return out, None


# ─── 나이 데이터 로더 ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_age_for_dataset(ds_name: str, ds_cfg_json: str) -> np.ndarray | None:
    cfg = json.loads(ds_cfg_json) if ds_cfg_json else {}
    table_csv = cfg.get("table_csv")
    age_col = cfg.get("age_col", "age")
    scale = float(cfg.get("age_scale", 1.0))
    if table_csv and Path(table_csv).exists():
        try:
            df = pd.read_csv(table_csv, low_memory=False)
            if age_col in df.columns:
                return pd.to_numeric(df[age_col], errors="coerce").to_numpy(dtype=float) * scale
        except Exception:
            pass
    hc = AGE_TABLE_HINTS.get(ds_name) or AGE_TABLE_HINTS.get(ds_name.lower())
    if hc:
        path, col, sc = hc
        try:
            df = pd.read_csv(path, low_memory=False)
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) * sc
        except Exception:
            pass
    return None


def parse_age_bins(text: str) -> list[tuple[float, float]]:
    """예: "0,18,30,50,70,200" → [(0,18),(18,30),(30,50),(50,70),(70,200)]."""
    raw = [t.strip() for t in text.replace(",", " ").split() if t.strip()]
    nums = []
    for t in raw:
        try:
            nums.append(float(t))
        except ValueError:
            continue
    nums = sorted(set(nums))
    return list(zip(nums[:-1], nums[1:]))


AGE_PALETTE = ["#CC79A7", "#9400D3", "#D55E00", "#F0E442",
               "#0072B2", "#009E73", "#E69F00", "#000000",
               "#56B4E9", "#882255"]


def assign_age_bin_labels(ages: np.ndarray, bins: list[tuple[float, float]]):
    """각 샘플의 bin index (없으면 -1) 반환."""
    out = np.full(len(ages), -1, dtype=np.int32)
    finite = np.isfinite(ages)
    for bi, (lo, hi) in enumerate(bins):
        mask = finite & (ages >= lo) & (ages < hi)
        out[mask] = bi
    return out


# ─── 1:1 균형 인덱스 ────────────────────────────────────────────────────────
def balanced_indices(binary: np.ndarray, seed: int = 42) -> np.ndarray:
    pos = np.where(binary == 1)[0]
    neg = np.where(binary == 0)[0]
    n = min(len(pos), len(neg))
    if n == 0:
        return np.arange(len(binary))
    rng = np.random.RandomState(seed)
    sel_p = rng.choice(pos, n, replace=False) if len(pos) > n else pos
    sel_n = rng.choice(neg, n, replace=False) if len(neg) > n else neg
    keep = np.concatenate([sel_p, sel_n])
    keep.sort()
    return keep


# ─── 논문용 Figure 렌더링 ────────────────────────────────────────────────────
def render_figure(
    selected_models, selected_datasets, selected_labels,
    all_data,          # {model: {dataset: (coords, emb_for_metrics, labels_raw, ages_or_None)}}
    umap_params,
    color_mode,        # "dataset" | "label" | "age"
    pos_neg_names,
    age_bins=None,     # list[(lo, hi)] for color_mode == "age"
    fig_dpi=200,
    point_size=4,
    point_alpha=0.5,
    title_prefix="",
):
    n_rows = len(selected_models)
    if color_mode == "dataset":
        n_cols = len(selected_datasets)
    elif color_mode == "label":
        n_cols = len(selected_labels)
    else:  # age
        n_cols = len(selected_datasets)
    if n_cols == 0 or n_rows == 0:
        return None, {}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "pdf.fonttype": 42,
    })

    fig = plt.figure(figsize=(max(5, 4.5 * n_cols), max(4, 4.0 * n_rows)),
                     dpi=fig_dpi)
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  hspace=0.35, wspace=0.15)

    metrics_summary = {}

    for r_idx, model_name in enumerate(selected_models):
        model_data = all_data.get(model_name, {})
        metrics_summary[model_name] = {}

        if color_mode == "dataset":
            for c_idx, ds_name in enumerate(selected_datasets):
                ax = fig.add_subplot(gs[r_idx, c_idx])
                entry = model_data.get(ds_name)
                if entry is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    _clean_ax(ax)
                    continue
                coords_ds = entry[0]

                for ds2 in selected_datasets:
                    if ds2 == ds_name:
                        continue
                    other = model_data.get(ds2)
                    if other is not None:
                        c2 = other[0]
                        ax.scatter(c2[:, 0], c2[:, 1], c="#DDDDDD",
                                   s=max(1, point_size - 1), alpha=0.25,
                                   rasterized=True, linewidths=0, zorder=1)

                ci = selected_datasets.index(ds_name)
                ax.scatter(coords_ds[:, 0], coords_ds[:, 1],
                           c=PALETTE[ci % len(PALETTE)],
                           marker=MARKERS[ci % len(MARKERS)],
                           s=point_size, alpha=point_alpha,
                           rasterized=True, linewidths=0, zorder=2,
                           label=f"{ds_name} (n={len(coords_ds):,})")

                ax.legend(fontsize=7, loc="upper right", framealpha=0.85,
                          handlelength=1.2, handletextpad=0.5, markerscale=2)

                if r_idx == 0:
                    ax.set_title(ds_name, fontweight="bold", pad=6)
                if c_idx == 0:
                    ax.set_ylabel(model_name, fontweight="bold", fontsize=10)
                else:
                    ax.set_ylabel("")
                _clean_ax(ax)

        elif color_mode == "label":
            for c_idx, (ds_name, lbl_name) in enumerate(selected_labels):
                ax = fig.add_subplot(gs[r_idx, c_idx])
                entry = model_data.get(ds_name)
                if entry is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    _clean_ax(ax)
                    continue
                coords_ds, emb_norm, labels_raw = entry[0], entry[1], entry[2]
                if labels_raw is None:
                    ax.text(0.5, 0.5, "No labels", ha="center", va="center")
                    _clean_ax(ax)
                    continue
                info = pos_neg_names.get((ds_name, lbl_name))
                if info is None:
                    ax.text(0.5, 0.5, "Label N/A", ha="center", va="center")
                    _clean_ax(ax)
                    continue
                lbl_idx, pos_name, neg_name = info
                if lbl_idx >= labels_raw.shape[1]:
                    ax.text(0.5, 0.5, "Idx OOB", ha="center", va="center")
                    _clean_ax(ax)
                    continue

                binary = (labels_raw[:, lbl_idx] > 0).astype(int)
                pos_c, neg_c = "#2ecc71", "#e74c3c"
                pos_mask = binary == 1
                neg_mask = ~pos_mask
                ax.scatter(coords_ds[neg_mask, 0], coords_ds[neg_mask, 1],
                           c=neg_c, s=point_size, alpha=point_alpha,
                           rasterized=True, linewidths=0, zorder=1)
                ax.scatter(coords_ds[pos_mask, 0], coords_ds[pos_mask, 1],
                           c=pos_c, s=point_size, alpha=point_alpha,
                           rasterized=True, linewidths=0, zorder=2)

                n_pos = int(binary.sum())
                n_neg = len(binary) - n_pos
                sil, bacc = compute_metrics(emb_norm, binary)
                metrics_summary[model_name][f"{ds_name}/{lbl_name}"] = {
                    "silhouette": sil, "knn5_bacc": bacc
                }

                sil_str = f"{sil:.3f}" if np.isfinite(sil) else "—"
                bacc_str = f"{bacc:.3f}" if np.isfinite(bacc) else "—"
                ax.set_xlabel(
                    f"sil={sil_str}  kNN-BACC={bacc_str}",
                    fontsize=8, labelpad=3
                )

                legend_els = [
                    mpatches.Patch(facecolor=pos_c,
                                   label=f"{pos_name} (n={n_pos:,})"),
                    mpatches.Patch(facecolor=neg_c,
                                   label=f"{neg_name} (n={n_neg:,})"),
                ]
                ax.legend(handles=legend_els, fontsize=7,
                          loc="upper right", framealpha=0.85,
                          handlelength=1.2, handletextpad=0.5)

                col_title = f"{ds_name} / {lbl_name}"
                if r_idx == 0:
                    ax.set_title(col_title, fontweight="bold", pad=6, fontsize=10)
                if c_idx == 0:
                    ax.set_ylabel(model_name, fontweight="bold", fontsize=10)
                else:
                    ax.set_ylabel("")
                _clean_ax(ax)

        else:  # color_mode == "age"
            for c_idx, ds_name in enumerate(selected_datasets):
                ax = fig.add_subplot(gs[r_idx, c_idx])
                entry = model_data.get(ds_name)
                if entry is None or len(entry) < 4 or entry[3] is None:
                    ax.text(0.5, 0.5, "No age data", ha="center", va="center")
                    _clean_ax(ax)
                    continue
                coords_ds = entry[0]
                ages = entry[3]
                bins = age_bins or []
                bin_idx = assign_age_bin_labels(ages, bins)

                # 무효(NaN/범위 밖) 회색
                invalid = bin_idx < 0
                if invalid.any():
                    ax.scatter(coords_ds[invalid, 0], coords_ds[invalid, 1],
                               c="#DDDDDD", s=max(1, point_size - 1),
                               alpha=0.25, rasterized=True, linewidths=0, zorder=1)
                legend_els = []
                for bi, (lo, hi) in enumerate(bins):
                    mask = bin_idx == bi
                    if not mask.any():
                        continue
                    color = AGE_PALETTE[bi % len(AGE_PALETTE)]
                    ax.scatter(coords_ds[mask, 0], coords_ds[mask, 1],
                               c=color, s=point_size, alpha=point_alpha,
                               rasterized=True, linewidths=0, zorder=2 + bi)
                    legend_els.append(mpatches.Patch(
                        facecolor=color,
                        label=f"{int(lo)}–{int(hi) if hi < 200 else '+'} (n={int(mask.sum()):,})"
                    ))
                if legend_els:
                    ax.legend(handles=legend_els, fontsize=6, loc="upper right",
                              framealpha=0.85, handlelength=1.2, handletextpad=0.5,
                              ncol=1)

                if r_idx == 0:
                    ax.set_title(ds_name, fontweight="bold", pad=6)
                if c_idx == 0:
                    ax.set_ylabel(model_name, fontweight="bold", fontsize=10)
                else:
                    ax.set_ylabel("")
                _clean_ax(ax)

    umap_str = (f"n_neighbors={umap_params['n_neighbors']}, "
                f"min_dist={umap_params['min_dist']}, "
                f"metric={umap_params['metric']}, "
                f"L2={'on' if umap_params.get('do_l2') else 'off'}")
    title = f"{title_prefix}({umap_str})" if title_prefix else f"ECG FM UMAP  ({umap_str})"
    fig.suptitle(title, fontsize=12, y=1.01, fontweight="bold")
    return fig, metrics_summary


def _clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_to_bytes(fig, fmt="png", dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════════════
# Streamlit App
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ECG FM · UMAP Explorer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .title-block {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid #30305a;
    border-radius: 12px;
    padding: 20px 28px 16px;
    margin-bottom: 20px;
  }
  .title-block h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; color: #e0e0ff;
    margin: 0 0 4px 0; letter-spacing: -0.5px;
  }
  .title-block p { color: #8888bb; font-size: 0.85rem; margin: 0; }
  .metric-card {
    background: #0f0f1a; border: 1px solid #2a2a4a;
    border-radius: 8px; padding: 10px 14px; margin: 4px 0;
    font-family: 'Space Mono', monospace; font-size: 0.78rem; color: #aaaacc;
  }
  .metric-card b { color: #ccccff; }
  div[data-testid="stSidebar"] { background: #0d0d1c; }
  div[data-testid="stSidebar"] label { color: #aaaacc !important; font-size: 0.82rem; }
  div[data-testid="stSidebar"] .stCheckbox { margin-bottom: 2px; }
  .stButton > button {
    width: 100%; border-radius: 6px;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
  }
  .badge {
    display: inline-block; background: #1e1e3a; border: 1px solid #3a3a6a;
    border-radius: 4px; padding: 1px 8px;
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    color: #8888ee; margin: 2px;
  }
  .preset-badge {
    display: inline-block; background: #14302a; border: 1px solid #2c6a55;
    border-radius: 4px; padding: 2px 10px;
    font-family: 'Space Mono', monospace; font-size: 0.74rem;
    color: #8fe5c8; margin: 2px 4px 2px 0;
  }
  .cache-badge {
    display: inline-block; background: #1f2a3e; border: 1px solid #4477AA;
    border-radius: 4px; padding: 2px 10px;
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    color: #b8d4ff; margin: 2px 4px 2px 0;
  }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="title-block">
  <h1>🫀 ECG FM · UMAP Explorer</h1>
  <p>결과 디렉토리: <code>{RESULT_DIR}</code> &nbsp;·&nbsp; 임베딩: <code>{EMB_DIR}</code></p>
</div>
""", unsafe_allow_html=True)

if not EMB_DIR.exists():
    st.error(f"❌ embeddings 디렉토리를 찾을 수 없습니다: `{EMB_DIR}`")
    st.stop()

all_models, all_datasets, model_info_map = discover_models_and_datasets(EMB_DIR)
# 기본 제외
EXCLUDED_DATASETS_DEFAULT = {"mimic4"}
EXCLUDED_MODELS_DEFAULT = {"MERL (ViT)", "MERL (ResNet)", "ECG-FM-KED", "HuBERT-ECG", "ST-MEM"}
all_datasets = [d for d in all_datasets if d not in EXCLUDED_DATASETS_DEFAULT]
all_models = [m for m in all_models if m not in EXCLUDED_MODELS_DEFAULT]
if not all_models or not all_datasets:
    st.error("임베딩 파일을 찾지 못했습니다.")
    st.stop()

# config.json 로드 (있으면)
ds_label_info: dict = {}
for cand in (RESULT_DIR / "config.json", EMB_DIR.parent / "config.json"):
    if cand.exists():
        try:
            with open(cand) as f:
                run_cfg = json.load(f)
            for ds_cfg in run_cfg.get("datasets", []) or []:
                ds_label_info[ds_cfg["name"]] = ds_cfg
        except Exception as e:
            st.warning(f"config.json 로드 실패: {cand} ({e})")
        break

# label_cols 추론
ds_to_label_cols: dict[str, list[str]] = {}
for ds_name in all_datasets:
    n_cols = 0
    for m_name in all_models:
        safe = model_info_map[m_name]["safe"]
        p = EMB_DIR / f"{safe}_{ds_name}_labels.npy"
        if p.exists():
            try:
                arr = np.load(p, mmap_mode="r")
                n_cols = arr.shape[1] if arr.ndim == 2 else (1 if arr.ndim == 1 else 0)
            except Exception:
                pass
            break
    if n_cols == 0:
        continue
    ds_to_label_cols[ds_name] = find_label_cols(ds_name, n_cols, ds_label_info.get(ds_name))


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 모델 선택")
    selected_models = []
    for m in all_models:
        fd = model_info_map[m].get("feature_dim")
        suffix = f"  (d={fd})" if fd else ""
        if st.checkbox(f"{m}{suffix}", value=True, key=f"model_{m}"):
            selected_models.append(m)

    st.divider()
    st.markdown("### 📁 데이터셋 선택")
    selected_datasets = []
    for ds in all_datasets:
        if st.checkbox(ds, value=True, key=f"ds_{ds}"):
            selected_datasets.append(ds)

    st.divider()
    st.markdown("### 🎯 비교할 UMAP 프리셋")
    selected_preset_names: list[str] = []
    for pname, pcfg in PRESETS.items():
        if st.checkbox(pname, value=True, key=f"preset_{pcfg['tag']}",
                       help=pcfg["desc"]):
            selected_preset_names.append(pname)
    use_user_preset = st.checkbox("사용자 설정 프리셋 추가", value=False,
                                  key="preset_user")

    st.divider()
    st.markdown("### 🏷️ 레이블 (선택)")
    available_labels = []
    for ds_name in all_datasets:
        for i, col in enumerate(ds_to_label_cols.get(ds_name, [])):
            available_labels.append((ds_name, col, i))

    selected_labels: list[tuple[str, str]] = []
    if available_labels:
        with st.expander(f"레이블 ({len(available_labels)}개)",
                         expanded=len(available_labels) <= 12):
            for ds_name, col, i in available_labels:
                default = (i == 0)
                key = f"lbl_{sanitize(ds_name)}__{sanitize(col)}_{i}"
                if st.checkbox(f"{ds_name} / {col}", value=default, key=key):
                    selected_labels.append((ds_name, col))
    else:
        st.caption("라벨 없음")

    st.divider()
    st.markdown("### 🎛️ 색상 기준")
    color_mode = st.radio("색상 기준", ["데이터셋", "레이블", "나이대"],
                          horizontal=True, label_visibility="collapsed")
    color_mode_key = {"데이터셋": "dataset", "레이블": "label",
                      "나이대": "age"}[color_mode]

    # 라벨 1:1 균형 (label 모드 전용)
    if color_mode_key == "label":
        balance_one_to_one = st.checkbox(
            "라벨 1:1 균형 다운샘플", value=False,
            help="각 (모델·데이터셋·라벨) 셀에서 양성/음성을 1:1로 다운샘플 후 UMAP 재계산. "
                 "캐시 사용 안 함."
        )
    else:
        balance_one_to_one = False

    # 나이 bin 편집 (age 모드 전용)
    if color_mode_key == "age":
        age_bins_text = st.text_area(
            "나이 bin 경계 (쉼표/공백 구분)",
            value="0, 18, 30, 40, 50, 60, 70, 80, 200",
            height=68,
            help="예: 0,18,30,50,70,200 → [0,18), [18,30), [30,50), [50,70), [70,200)",
        )
        AGE_BINS = parse_age_bins(age_bins_text)
        if not AGE_BINS:
            st.warning("유효한 나이 bin이 없습니다.")
    else:
        AGE_BINS = []

    if use_user_preset:
        st.divider()
        st.markdown("### ⚙️ 사용자 프리셋 파라미터")
        u_n_neighbors = st.slider("n_neighbors", 5, 100, 30, 5)
        u_min_dist    = st.slider("min_dist", 0.0, 0.9, 0.05, 0.05)
        u_metric      = st.selectbox("metric", ["cosine", "euclidean", "correlation"], index=0)
        u_do_l2       = st.checkbox("L2 정규화", value=True)
    else:
        u_n_neighbors = u_min_dist = None
        u_metric = "cosine"
        u_do_l2 = True

    st.divider()
    st.markdown("### 🌱 공통 옵션")
    seed = st.number_input("random_seed", 0, 9999, 42)
    use_full = st.checkbox("전체 샘플 사용", value=True,
                           help="OFF시 데이터셋당 n_samples로 다운샘플 (캐시 무시)")
    if not use_full:
        n_samples = st.slider("샘플 수 (데이터셋 당)",
                              min_value=500, max_value=50000, value=5000, step=500)
    else:
        n_samples = 10**9

    point_size  = st.slider("점 크기", 1, 15, 4)
    point_alpha = st.slider("투명도", 0.1, 1.0, 0.5, 0.05)
    export_dpi  = st.select_slider("저장 DPI", [150, 200, 300], value=300)

    st.divider()
    st.markdown("### 🛠️ 캐시")
    if st.button("🔄 모든 (모델 × 프리셋) UMAP 캐시 사전 생성"):
        st.session_state["_run_precache"] = True


# 사용자 프리셋 추가 (선택 시)
if use_user_preset:
    PRESETS["사용자 설정"] = {
        "tag": "user",
        "n_neighbors": int(u_n_neighbors),
        "min_dist": float(u_min_dist),
        "metric": u_metric,
        "do_l2": bool(u_do_l2),
        "desc": "사용자 정의 (캐시 사용/저장 안 함)",
    }
    selected_preset_names.append("사용자 설정")

# ─── 입력 검증 ───────────────────────────────────────────────────────────────
if not selected_models:
    st.warning("최소 1개 이상의 모델을 선택하세요."); st.stop()
if not selected_datasets:
    st.warning("최소 1개 이상의 데이터셋을 선택하세요."); st.stop()
if not selected_preset_names:
    st.warning("최소 1개 이상의 프리셋을 선택하세요."); st.stop()
if color_mode_key == "label" and not selected_labels:
    st.warning("레이블 모드에서는 최소 1개 이상의 레이블을 선택하세요."); st.stop()

# ─── 헤더 배지 ───────────────────────────────────────────────────────────────
ordered_ds = sorted(selected_datasets)  # 캐시 일관성을 위해 알파벳 순 (ptbxl, zzu)

badge_html = "".join(f'<span class="badge">{m}</span>' for m in selected_models)
badge_html += " &nbsp;×&nbsp; "
badge_html += "".join(f'<span class="badge">{d}</span>' for d in ordered_ds)
badge_html += "<br/>"
badge_html += "".join(f'<span class="preset-badge">▶ {p}</span>'
                      for p in selected_preset_names)
st.markdown(badge_html, unsafe_allow_html=True)
st.caption("")

# ─── 모델별 임베딩 일괄 로드 (재사용) ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_for_model(emb_dir_str: str, safe: str, ds_tuple: tuple,
                       n_cap: int):
    out_emb, out_lbl = {}, {}
    for d in ds_tuple:
        e, l = load_embeddings(emb_dir_str, safe, d)
        if e is None:
            continue
        if n_cap < len(e):
            e2, l2 = subsample(e, l, n_cap)
            out_emb[d] = e2; out_lbl[d] = l2
        else:
            out_emb[d] = e; out_lbl[d] = l
    return out_emb, out_lbl


# ─── 캐시 사전 생성 (요청 시) ────────────────────────────────────────────────
if st.session_state.get("_run_precache"):
    st.session_state["_run_precache"] = False
    pcache_targets = [(m, p) for m in all_models
                      for p in PRESETS.values()
                      if p["tag"] in {"orig", "cosineL2", "euclideanL2"}]
    bar = st.progress(0, text="캐시 생성 중...")
    log_lines = []
    for i, (m_name, preset) in enumerate(pcache_targets):
        safe = model_info_map[m_name]["safe"]
        emb_by_ds, _ = load_all_for_model(str(EMB_DIR), safe,
                                          tuple(all_datasets), 10**9)
        if not emb_by_ds:
            log_lines.append(f"  ⏭ {m_name}: 임베딩 없음")
            continue
        ordered_all = sorted(emb_by_ds.keys())
        _, save_fn = cache_filenames_for(safe, preset["tag"], tuple(ordered_all))
        target = EMB_DIR / save_fn if save_fn else None
        if target is not None and target.exists():
            log_lines.append(f"  ✅ {m_name} / {preset['tag']}: 이미 존재 ({target.name})")
        else:
            bar.progress(i / max(len(pcache_targets), 1),
                         text=f"{m_name} / {preset['tag']} 계산 중...")
            compute_combined_for_model(EMB_DIR, safe, ordered_all,
                                       emb_by_ds, preset, int(seed),
                                       try_cache=False, save_after_compute=True)
            log_lines.append(f"  💾 {m_name} / {preset['tag']}: 저장 → {target.name}")
    bar.empty()
    st.success("캐시 사전 생성 완료")
    with st.expander("로그", expanded=False):
        st.code("\n".join(log_lines))


# ─── 메인 파이프라인: 프리셋별로 all_data 채우기 ───────────────────────────
@st.cache_data(show_spinner=False)
def hydrate_preset(emb_dir_str: str, safes: tuple, ordered_ds_tuple: tuple,
                   preset_key: tuple, seed: int, n_cap: int,
                   balance_key: tuple = ()):
    """
    preset_key = (tag, n_neighbors, min_dist, metric, do_l2)
    balance_key = () or (label_idx_per_ds_tuple) — 1:1 다운샘플 시 cache buster 역할
    """
    (tag, nn, md, metric, do_l2) = preset_key
    preset = {"tag": tag, "n_neighbors": nn, "min_dist": md,
              "metric": metric, "do_l2": do_l2}
    coords_per_model: dict[str, dict] = {}
    cache_hits: dict[str, str | None] = {}
    for m_name, safe in safes:
        emb_by_ds, _ = load_all_for_model(emb_dir_str, safe,
                                          ordered_ds_tuple, n_cap)
        present = [d for d in ordered_ds_tuple if d in emb_by_ds]
        if not present:
            coords_per_model[m_name] = {}
            cache_hits[m_name] = None
            continue
        # 1:1 균형이 켜져 있으면 임베딩이 변하므로 캐시 사용/저장 모두 OFF
        is_balanced = bool(balance_key)
        try_cache = (tag in {"orig", "cosineL2", "euclideanL2"} and n_cap >= 10**8
                     and not is_balanced)
        save_after = try_cache
        coords_by_ds, hit = compute_combined_for_model(
            Path(emb_dir_str), safe, present, emb_by_ds,
            preset, seed, try_cache=try_cache, save_after_compute=save_after,
        )
        coords_per_model[m_name] = coords_by_ds
        cache_hits[m_name] = hit
    return coords_per_model, cache_hits


def _apply_balance_filter(safe: str, emb_by_ds: dict, lbl_by_ds: dict,
                          ages_by_ds: dict):
    """선택된 (ds, label)에 대해 1:1 균형 인덱스를 만들고 모든 배열을 필터링."""
    new_emb, new_lbl, new_ages = {}, {}, {}
    for d in emb_by_ds:
        # 첫 번째 selected_label that matches this dataset
        target = None
        for (ds_n, lbl_n) in selected_labels:
            if ds_n == d:
                lbl_idx = ds_to_label_cols.get(d, []).index(lbl_n) \
                    if lbl_n in ds_to_label_cols.get(d, []) else None
                if lbl_idx is None:
                    continue
                target = lbl_idx
                break
        l = lbl_by_ds.get(d)
        if target is None or l is None or l.ndim != 2 or target >= l.shape[1]:
            new_emb[d] = emb_by_ds[d]; new_lbl[d] = l
            new_ages[d] = ages_by_ds.get(d)
            continue
        binary = (l[:, target] > 0).astype(int)
        keep = balanced_indices(binary)
        new_emb[d] = emb_by_ds[d][keep]
        new_lbl[d] = l[keep]
        a = ages_by_ds.get(d)
        new_ages[d] = a[keep] if a is not None else None
    return new_emb, new_lbl, new_ages


def build_all_data(preset_name: str):
    preset = PRESETS[preset_name]
    safes = tuple((m, model_info_map[m]["safe"]) for m in selected_models)
    ds_tuple = tuple(ordered_ds)
    preset_key = (preset["tag"], preset["n_neighbors"], preset["min_dist"],
                  preset["metric"], preset["do_l2"])

    # 1:1 균형이 켜져 있으면 별도 경로로 처리 (캐시 미사용)
    if balance_one_to_one and color_mode_key == "label" and selected_labels:
        all_data: dict[str, dict] = {}
        cache_hits: dict[str, str | None] = {m: None for m in selected_models}
        for m_name in selected_models:
            safe = model_info_map[m_name]["safe"]
            emb_by_ds, lbl_by_ds = load_all_for_model(str(EMB_DIR), safe,
                                                      ds_tuple, int(n_samples))
            ages_by_ds = {d: load_age_for_dataset(
                              d, json.dumps(ds_label_info.get(d, {}), default=str))
                          for d in emb_by_ds}
            # 임베딩 길이에 맞춰 ages를 trim
            for d in list(ages_by_ds.keys()):
                a = ages_by_ds[d]
                if a is not None:
                    ages_by_ds[d] = a[:len(emb_by_ds[d])] if len(a) >= len(emb_by_ds[d]) \
                        else np.concatenate([a, np.full(len(emb_by_ds[d]) - len(a), np.nan)])
            emb_by_ds, lbl_by_ds, ages_by_ds = _apply_balance_filter(
                safe, emb_by_ds, lbl_by_ds, ages_by_ds)
            present = [d for d in ds_tuple if d in emb_by_ds and len(emb_by_ds[d]) > 0]
            if not present:
                all_data[m_name] = {}
                continue
            coords_by_ds, _ = compute_combined_for_model(
                EMB_DIR, safe, present, emb_by_ds, preset, int(seed),
                try_cache=False, save_after_compute=False,
            )
            all_data[m_name] = {}
            for d in present:
                emb = emb_by_ds[d].astype(np.float32)
                metrics_emb = l2_normalize(emb) if preset["do_l2"] else emb
                all_data[m_name][d] = (coords_by_ds[d], metrics_emb,
                                        lbl_by_ds.get(d), ages_by_ds.get(d))
        return all_data, cache_hits, preset

    # 기본 경로 (캐시 사용)
    coords_per_model, cache_hits = hydrate_preset(
        str(EMB_DIR), safes, ds_tuple, preset_key, int(seed), int(n_samples)
    )

    all_data: dict[str, dict] = {}
    for m_name in selected_models:
        safe = model_info_map[m_name]["safe"]
        emb_by_ds, lbl_by_ds = load_all_for_model(str(EMB_DIR), safe,
                                                  ds_tuple, int(n_samples))
        all_data[m_name] = {}
        coords_by_ds = coords_per_model.get(m_name, {})
        for d in ordered_ds:
            if d not in coords_by_ds or d not in emb_by_ds:
                continue
            emb = emb_by_ds[d].astype(np.float32)
            metrics_emb = l2_normalize(emb) if preset["do_l2"] else emb
            ages = None
            if color_mode_key == "age":
                a = load_age_for_dataset(
                    d, json.dumps(ds_label_info.get(d, {}), default=str))
                if a is not None:
                    if len(a) >= len(emb):
                        ages = a[:len(emb)]
                    else:
                        ages = np.concatenate(
                            [a, np.full(len(emb) - len(a), np.nan)])
            all_data[m_name][d] = (coords_by_ds[d], metrics_emb,
                                    lbl_by_ds.get(d), ages)
    return all_data, cache_hits, preset


# pos_neg_names 구성
pos_neg_names = {}
for ds_name in selected_datasets:
    info = ds_label_info.get(ds_name, {})
    pos_name = info.get("positive_label", "Positive")
    neg_name = info.get("negative_label", "Negative")
    for i, col in enumerate(ds_to_label_cols.get(ds_name, [])):
        if (ds_name, col) in selected_labels:
            pos_neg_names[(ds_name, col)] = (i, pos_name, neg_name)


# ─── 프리셋별 탭 렌더 ────────────────────────────────────────────────────────
def render_preset_tab(preset_name: str):
    with st.spinner(f"[{preset_name}] 임베딩 로드 & UMAP 준비..."):
        all_data, cache_hits, preset = build_all_data(preset_name)

    # 캐시 히트 배지
    n_hit = sum(1 for v in cache_hits.values() if v)
    n_total = len(cache_hits)
    if n_total > 0:
        if n_hit == n_total:
            st.markdown(f'<span class="cache-badge">✅ 모든 {n_total}개 모델 캐시 히트</span>',
                        unsafe_allow_html=True)
        elif n_hit > 0:
            st.markdown(
                f'<span class="cache-badge">✅ {n_hit}/{n_total} 캐시 히트 (나머지는 새로 계산)</span>',
                unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="cache-badge">🧮 캐시 없음 (전부 새로 계산)</span>',
                        unsafe_allow_html=True)

    umap_params = dict(
        n_neighbors=preset["n_neighbors"], min_dist=preset["min_dist"],
        metric=preset["metric"], do_l2=preset["do_l2"], seed=int(seed),
    )
    fig, metrics_summary = render_figure(
        selected_models=selected_models,
        selected_datasets=ordered_ds,
        selected_labels=[(d, l) for d, l in selected_labels if d in ordered_ds]
                        if color_mode_key == "label" else [],
        all_data=all_data,
        umap_params=umap_params,
        color_mode=color_mode_key,
        pos_neg_names=pos_neg_names,
        age_bins=AGE_BINS,
        fig_dpi=100,
        point_size=point_size,
        point_alpha=point_alpha,
        title_prefix=f"[{preset['tag']}{' bal' if balance_one_to_one and color_mode_key=='label' else ''}] ",
    )
    if fig is None:
        st.warning("표시할 데이터가 없습니다.")
        return None, None, None

    st.pyplot(fig, width="stretch")

    # 메트릭 요약
    if metrics_summary and any(metrics_summary.values()):
        st.markdown("##### 📊 정량 지표")
        cols = st.columns(min(len(selected_models), 4))
        for i, (m_name, mets) in enumerate(metrics_summary.items()):
            with cols[i % len(cols)]:
                st.markdown(f"**{m_name}**")
                if mets:
                    for label_key, scores in mets.items():
                        sil = scores.get("silhouette", float("nan"))
                        bacc = scores.get("knn5_bacc", float("nan"))
                        sil_str  = f"{sil:.4f}"  if np.isfinite(sil)  else "—"
                        bacc_str = f"{bacc:.4f}" if np.isfinite(bacc) else "—"
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<b>{label_key}</b><br>'
                            f'Silhouette: {sil_str}<br>'
                            f'kNN-BACC: {bacc_str}'
                            f'</div>', unsafe_allow_html=True)
                else:
                    st.caption("레이블 지표 없음")

    return fig, all_data, umap_params


# 프리셋 1개면 단일 화면, 2개 이상이면 탭으로
if len(selected_preset_names) == 1:
    pname = selected_preset_names[0]
    st.markdown(f"#### {pname}")
    fig_main, _, umap_params_main = render_preset_tab(pname)
else:
    tabs = st.tabs([f"📊 {p}" for p in selected_preset_names])
    fig_main = None
    umap_params_main = None
    for tab, pname in zip(tabs, selected_preset_names):
        with tab:
            f, _, up = render_preset_tab(pname)
            if fig_main is None:
                fig_main, umap_params_main = f, up


# ─── 다운로드 ───────────────────────────────────────────────────────────────
if fig_main is not None:
    st.markdown("---")
    st.markdown("#### 💾 첫 프리셋 이미지 저장")
    dl1, dl2, dl3 = st.columns([1, 1, 2])
    with dl1:
        png_bytes = fig_to_bytes(fig_main, fmt="png", dpi=export_dpi)
        st.download_button(f"⬇️ PNG ({export_dpi} DPI)", png_bytes,
                           "ecg_fm_umap.png", "image/png", width="stretch")
    with dl2:
        pdf_bytes = fig_to_bytes(fig_main, fmt="pdf", dpi=export_dpi)
        st.download_button("⬇️ PDF (벡터)", pdf_bytes,
                           "ecg_fm_umap.pdf", "application/pdf", width="stretch")
    with dl3:
        caption_text = (
            f"UMAP projections of ECG embeddings from {len(selected_models)} foundation model(s) "
            f"across {len(ordered_ds)} dataset(s). "
            f"UMAP: n_neighbors={umap_params_main['n_neighbors']}, "
            f"min_dist={umap_params_main['min_dist']}, metric={umap_params_main['metric']}, "
            f"L2={'on' if umap_params_main.get('do_l2') else 'off'}, seed={seed}. "
            f"Colors indicate "
            f"{'dataset origin' if color_mode_key == 'dataset' else 'diagnostic category'}."
        )
        st.text_area("📝 Figure Caption", caption_text, height=90)

plt.close("all")
