"""
ECG Foundation Model — Notebook-friendly UMAP View
====================================================
Jupyter 노트북에서 빠르게 호출해 UMAP 비교 그림을 그려보기 위한 모듈.
streamlit 의존성 없음. `umap_explorer.py`와 동일한 캐시 파일을 공유.

사용 예 (노트북에서):

    from scripts.umap_view import Explorer

    ex = Explorer("results/embeddings")
    ex.print_status()                                    # 가용 모델/데이터셋 출력
    ex.view(color_mode="dataset")                        # 기본: 모든 모델 × 모든 데이터셋
    ex.view(models=["CPC","ECG-FM"], datasets=["ptbxl","zzu"], preset="cosineL2")
    ex.compare(presets=["orig","cosineL2"])              # 두 프리셋 나란히
    ex.view(color_mode="age", age_bins="0,18,40,60,80,200")
    ex.view(color_mode="label", labels=[("ptbxl","NORM")], balance_one_to_one=True)
    ex.interact()                                        # ipywidgets UI
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from umap import UMAP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = PROJECT_ROOT / "labels"
# 진단 라벨은 ../benchmark/labels/ 의 paper 라벨 csv 우선 사용
PAPER_LABELS_DIR = Path("/home/irteam/local-node-d/tykim/benchmark/labels")

# ─── 상수 ───────────────────────────────────────────────────────────────────
PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44",
           "#66CCEE", "#AA3377", "#BBBBBB", "#332288"]
MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]

MODEL_ORDER_HINT = [
    "CPC", "ECG-FM", "ECG-FM-KED", "ECG-Founder", "ECG-JEPA",
    "HuBERT-ECG", "MERL (ResNet)", "MERL (ViT)", "ST-MEM",
]

# 기본 discover에서 제외할 데이터셋 (Explorer(exclude_datasets=...)로 override 가능)
DEFAULT_EXCLUDED_DATASETS: tuple[str, ...] = ("mimic4",)

# 기본 discover에서 제외할 모델 (override 가능)
# 사용자 요청: CPC, ECG-FM, ECG-Founder, ECG-JEPA 4개만 사용
DEFAULT_EXCLUDED_MODELS: tuple[str, ...] = (
    "MERL (ViT)", "MERL (ResNet)", "ECG-FM-KED", "HuBERT-ECG", "ST-MEM",
)

# 데이터셋 → paper label csv (PAPER_LABELS_DIR 또는 LABELS_DIR 에서 검색)
LABEL_CSV_HINTS = {
    "ptbxl":   "ptbxl_super_paper_labels.csv",
    "zzu":     "zzu_paper_labels.csv",
    "chapman": "chapman_paper_labels.csv",
    "sph":     "sph_paper_labels.csv",
    "code15":  "code15_paper_labels.csv",
    # mimic4 — paper label 없음
    "PTB-XL":  "ptbxl_super_paper_labels.csv",
    "ZZU-pECG": "zzu_paper_labels.csv",
}

# 데이터셋 → table csv (filepath 정렬 join용)
DATASET_TABLE_HINTS = {
    "ptbxl":   "/home/irteam/ddn-opendata1/h5/physionet/v2.0/ptbxl_table.csv",
    "zzu":     "/home/irteam/ddn-opendata1/h5/ZZU-pECG/v2.0/ecg_table.csv",
    "chapman": "/home/irteam/ddn-opendata1/h5/physionet/v2.0/chapman_table.csv",
    "sph":     "/home/irteam/ddn-opendata1/h5/sph/v2.0/ecg_table.csv",
    "code15":  "/home/irteam/ddn-opendata1/h5/code15/v2.0/code15_table.csv",
    "mimic4":  "/home/irteam/ddn-opendata1/h5/mimic4/v2.0/mimic4_table.csv",
}

AGE_TABLE_HINTS = {
    ds: (path, "age", 100.0) for ds, path in DATASET_TABLE_HINTS.items()
}


def _resolve_label_csv(filename: str) -> Optional[Path]:
    """PAPER_LABELS_DIR / LABELS_DIR 순으로 라벨 csv 탐색."""
    if not filename:
        return None
    for d in (PAPER_LABELS_DIR, LABELS_DIR):
        p = d / filename
        if p.exists():
            return p
    return None

AGE_PALETTE = ["#CC79A7", "#9400D3", "#D55E00", "#F0E442",
               "#0072B2", "#009E73", "#E69F00", "#000000",
               "#56B4E9", "#882255"]

# 진단용 categorical 팔레트 — matplotlib tab10 + tab20 보강 (서로 명확히 구분)
DX_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

PRESETS = {
    "orig": dict(tag="orig",
                 n_neighbors=15, min_dist=0.1, metric="euclidean", do_l2=False,
                 desc="euclidean + L2 OFF (run_all_embedding_umap.py와 동일)"),
    "euclideanL2": dict(tag="euclideanL2",
                        n_neighbors=15, min_dist=0.1, metric="euclidean", do_l2=True,
                        desc="euclidean + L2 ON  (orig 과 L2만 토글한 ablation 짝)"),
    "cosineL2": dict(tag="cosineL2",
                     n_neighbors=30, min_dist=0.05, metric="cosine", do_l2=True,
                     desc="cosine + L2 (foundation model 비교용)"),
}


# ─── 유틸 ────────────────────────────────────────────────────────────────────
def sanitize(name: str) -> str:
    return (name.replace(" ", "_")
                .replace("(", "").replace(")", "").replace("/", "_"))


def cache_filenames_for(safe: str, preset_tag: str, ds_tuple: tuple) -> tuple[tuple, str]:
    is_legacy = tuple(ds_tuple) == ("ptbxl", "zzu")
    suffix = "" if is_legacy else "__" + "_".join(ds_tuple)
    if preset_tag == "orig":
        save_fn = f"{safe}_umap_coords{suffix}.npy"
        loads = [save_fn]
        if is_legacy:
            loads.append(f"{safe}_umap_coords_all.npy")
        return tuple(loads), save_fn
    if preset_tag == "cosineL2":
        save_fn = f"{safe}_umap_coords_cosineL2{suffix}.npy"
        return (save_fn,), save_fn
    if preset_tag == "euclideanL2":
        save_fn = f"{safe}_umap_coords_euclideanL2{suffix}.npy"
        return (save_fn,), save_fn
    return (), ""


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
        display = next((m for m in MODEL_ORDER_HINT if sanitize(m) == safe), safe)
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
        matched = None
        for safe in sorted(known_safes, key=len, reverse=True):
            if stem.startswith(safe + "_"):
                matched = safe; break
        if matched is None:
            for hint in sorted(MODEL_ORDER_HINT, key=lambda x: len(sanitize(x)), reverse=True):
                s = sanitize(hint)
                if stem.startswith(s + "_"):
                    matched = s
                    if hint not in models:
                        models[hint] = {"safe": s, "feature_dim": None}
                    known_safes.add(s); break
        if matched is None:
            continue
        datasets.add(stem[len(matched) + 1:])

    sorted_models = sorted(models.keys(),
                           key=lambda m: (MODEL_ORDER_HINT.index(m)
                                          if m in MODEL_ORDER_HINT else 999, m))
    return sorted_models, sorted(datasets), models


_LABEL_SKIP_COLS = {"filepath", "path", "file", "dataset",
                    "pid", "rid", "oid", "sid", "age", "sex"}


def _label_cols_from_csv(csv_path: Path) -> list[str]:
    cols = list(pd.read_csv(csv_path, nrows=0).columns)
    return [c for c in cols if c not in _LABEL_SKIP_COLS]


def find_label_cols(ds_name: str, fallback_n: int = 0,
                    config_info: Optional[dict] = None) -> list[str]:
    """paper label csv가 있으면 그 컬럼을, 없으면 config / fallback 사용."""
    if config_info and config_info.get("label_cols"):
        return list(config_info["label_cols"])
    csv_name = LABEL_CSV_HINTS.get(ds_name) or LABEL_CSV_HINTS.get(ds_name.lower())
    if csv_name:
        p = _resolve_label_csv(csv_name)
        if p:
            try:
                lc = _label_cols_from_csv(p)
                if lc:
                    return lc
            except Exception:
                pass
    if config_info and config_info.get("label_col"):
        return [config_info["label_col"]]
    return [f"label_{i}" for i in range(fallback_n)]


@lru_cache(maxsize=32)
def _load_paper_labels_aligned(ds_name: str, n_rows: int):
    """
    table_csv 행 순서대로 라벨 csv를 left-join 하여 처음 n_rows × K 매트릭스 반환.
    임베딩 추출 시 첫 n_rows 행만 사용했으므로 동일 인덱싱.
    return: (matrix [n_rows, K] float32 (NaN→0), label_cols list)
            없으면 (None, []).
    """
    csv_name = LABEL_CSV_HINTS.get(ds_name) or LABEL_CSV_HINTS.get(ds_name.lower())
    if not csv_name:
        return None, []
    label_csv = _resolve_label_csv(csv_name)
    if label_csv is None:
        return None, []
    table_csv = DATASET_TABLE_HINTS.get(ds_name) or DATASET_TABLE_HINTS.get(ds_name.lower())
    if not table_csv or not Path(table_csv).exists():
        return None, []
    try:
        tdf = pd.read_csv(table_csv, low_memory=False, usecols=lambda c: c == "filepath")
        ldf = pd.read_csv(label_csv, low_memory=False)
    except Exception:
        return None, []
    if "filepath" not in tdf.columns or "filepath" not in ldf.columns:
        return None, []
    label_cols = [c for c in ldf.columns if c not in _LABEL_SKIP_COLS]
    if not label_cols:
        return None, []
    merged = tdf[["filepath"]].merge(
        ldf[["filepath"] + label_cols].drop_duplicates(subset="filepath"),
        on="filepath", how="left",
    )
    take = merged.iloc[:n_rows]
    mat = take[label_cols].to_numpy(dtype=np.float32, na_value=0.0)
    return mat, label_cols


@lru_cache(maxsize=128)
def _load_embeddings_cached(emb_dir_str: str, model_safe: str, dataset_name: str):
    emb_dir = Path(emb_dir_str)
    emb_path = emb_dir / f"{model_safe}_{dataset_name}.npy"
    lbl_path = emb_dir / f"{model_safe}_{dataset_name}_labels.npy"
    if not emb_path.exists():
        return None, None
    emb = np.load(emb_path)
    lbl = np.load(lbl_path) if lbl_path.exists() else None
    return emb, lbl


@lru_cache(maxsize=64)
def _compute_umap_cached(emb_bytes: bytes, shape: tuple, n_neighbors: int,
                         min_dist: float, metric: str, seed: int):
    arr = np.frombuffer(emb_bytes, dtype=np.float32).reshape(shape)
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors,
                   min_dist=min_dist, metric=metric, random_state=seed)
    return reducer.fit_transform(arr)


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)


def subsample(emb, labels, n, seed=42):
    if n >= len(emb):
        return emb, labels
    idx = np.random.RandomState(seed).choice(len(emb), n, replace=False)
    return emb[idx], (labels[idx] if labels is not None else None)


def compute_metrics(emb, binary):
    if len(set(binary.tolist())) < 2:
        return float("nan"), float("nan")
    n = min(5000, len(emb))
    idx = np.random.RandomState(42).choice(len(emb), n, replace=False)
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


def try_load_cached_combined(emb_dir: Path, ordered_ds, sizes,
                             load_candidates):
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
            out[d] = coords[off:off + sizes[d]]
            off += sizes[d]
        return out, p.name
    return None, None


def save_cached_combined(emb_dir: Path, save_filename, ordered_ds, coords_by_ds):
    if not save_filename:
        return
    parts = [coords_by_ds[d] for d in ordered_ds]
    np.save(emb_dir / save_filename, np.concatenate(parts, axis=0))


def compute_combined_for_model(emb_dir, safe, ordered_ds, emb_by_ds, preset,
                               seed, try_cache=True, save_after=True):
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
    coords = _compute_umap_cached(concat.tobytes(), concat.shape,
                                  preset["n_neighbors"], float(preset["min_dist"]),
                                  preset["metric"], int(seed))
    out, off = {}, 0
    for d in ordered_ds:
        out[d] = coords[off:off + sizes[d]]
        off += sizes[d]
    if save_after and save_fn:
        try:
            save_cached_combined(emb_dir, save_fn, ordered_ds, out)
        except Exception:
            pass
    return out, None


@lru_cache(maxsize=32)
def _load_age_cached(ds_name: str, ds_cfg_json: str):
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


def parse_age_bins(text):
    if isinstance(text, (list, tuple)):
        nums = sorted(set(float(t) for t in text))
    else:
        raw = [t.strip() for t in str(text).replace(",", " ").split() if t.strip()]
        nums = sorted({float(t) for t in raw})
    return list(zip(nums[:-1], nums[1:]))


def assign_age_bin_labels(ages, bins):
    out = np.full(len(ages), -1, dtype=np.int32)
    finite = np.isfinite(ages)
    for bi, (lo, hi) in enumerate(bins):
        out[finite & (ages >= lo) & (ages < hi)] = bi
    return out


def balanced_indices(binary, seed=42):
    pos = np.where(binary == 1)[0]
    neg = np.where(binary == 0)[0]
    n = min(len(pos), len(neg))
    if n == 0:
        return np.arange(len(binary))
    rng = np.random.RandomState(seed)
    sel_p = rng.choice(pos, n, replace=False) if len(pos) > n else pos
    sel_n = rng.choice(neg, n, replace=False) if len(neg) > n else neg
    keep = np.concatenate([sel_p, sel_n]); keep.sort()
    return keep


def _clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ─── Figure 렌더링 ──────────────────────────────────────────────────────────
def render_figure(
    selected_models, selected_datasets, selected_labels,
    all_data, umap_params, color_mode, pos_neg_names,
    age_bins=None, fig_dpi=120, point_size=4, point_alpha=0.5,
    title_prefix="",
):
    n_rows = len(selected_models)
    if color_mode == "dataset":
        n_cols = len(selected_datasets)
    elif color_mode == "label":
        n_cols = len(selected_labels)
    else:  # age — 컬럼 = age bin 수
        n_cols = len(age_bins or [])
    if n_cols == 0 or n_rows == 0:
        return None, {}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(max(5, 4.5 * n_cols), max(4, 4.0 * n_rows)),
                     dpi=fig_dpi)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.15)
    metrics = {}

    for r, mname in enumerate(selected_models):
        md = all_data.get(mname, {})
        metrics[mname] = {}
        if color_mode == "dataset":
            for c, dname in enumerate(selected_datasets):
                ax = fig.add_subplot(gs[r, c])
                e = md.get(dname)
                if e is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    _clean_ax(ax); continue
                cds = e[0]
                for d2 in selected_datasets:
                    if d2 == dname: continue
                    o = md.get(d2)
                    if o is not None:
                        ax.scatter(o[0][:, 0], o[0][:, 1], c="#DDDDDD",
                                   s=max(1, point_size - 1), alpha=0.25,
                                   rasterized=True, linewidths=0, zorder=1)
                ci = selected_datasets.index(dname)
                ax.scatter(cds[:, 0], cds[:, 1],
                           c=PALETTE[ci % len(PALETTE)],
                           marker=MARKERS[ci % len(MARKERS)],
                           s=point_size, alpha=point_alpha,
                           rasterized=True, linewidths=0, zorder=2,
                           label=f"{dname} (n={len(cds):,})")
                ax.legend(fontsize=7, loc="upper right", framealpha=0.85,
                          handlelength=1.2, handletextpad=0.5, markerscale=2)
                if r == 0: ax.set_title(dname, fontweight="bold", pad=6)
                if c == 0: ax.set_ylabel(mname, fontweight="bold", fontsize=10)
                _clean_ax(ax)

        elif color_mode == "label":
            for c, (dname, lname) in enumerate(selected_labels):
                ax = fig.add_subplot(gs[r, c])
                e = md.get(dname)
                if e is None or e[2] is None:
                    ax.text(0.5, 0.5, "No labels", ha="center", va="center")
                    _clean_ax(ax); continue
                cds, emn, lbls = e[0], e[1], e[2]
                info = pos_neg_names.get((dname, lname))
                if info is None:
                    ax.text(0.5, 0.5, "Label N/A", ha="center", va="center")
                    _clean_ax(ax); continue
                lidx, pos_n, neg_n = info
                if lidx >= lbls.shape[1]:
                    ax.text(0.5, 0.5, "Idx OOB", ha="center", va="center")
                    _clean_ax(ax); continue
                bin_ = (lbls[:, lidx] > 0).astype(int)
                pos_c, neg_c = "#2ecc71", "#e74c3c"
                pmask = bin_ == 1
                ax.scatter(cds[~pmask, 0], cds[~pmask, 1], c=neg_c,
                           s=point_size, alpha=point_alpha,
                           rasterized=True, linewidths=0, zorder=1)
                ax.scatter(cds[pmask, 0], cds[pmask, 1], c=pos_c,
                           s=point_size, alpha=point_alpha,
                           rasterized=True, linewidths=0, zorder=2)
                np_ = int(bin_.sum()); nn_ = len(bin_) - np_
                sil, bacc = compute_metrics(emn, bin_)
                metrics[mname][f"{dname}/{lname}"] = {"silhouette": sil, "knn5_bacc": bacc}
                sil_s = f"{sil:.3f}" if np.isfinite(sil) else "—"
                bacc_s = f"{bacc:.3f}" if np.isfinite(bacc) else "—"
                ax.set_xlabel(f"sil={sil_s}  kNN-BACC={bacc_s}", fontsize=8, labelpad=3)
                legend_els = [
                    mpatches.Patch(facecolor=pos_c, label=f"{pos_n} (n={np_:,})"),
                    mpatches.Patch(facecolor=neg_c, label=f"{neg_n} (n={nn_:,})"),
                ]
                ax.legend(handles=legend_els, fontsize=7, loc="upper right",
                          framealpha=0.85, handlelength=1.2, handletextpad=0.5)
                if r == 0:
                    ax.set_title(f"{dname} / {lname}", fontweight="bold",
                                 pad=6, fontsize=10)
                if c == 0:
                    ax.set_ylabel(mname, fontweight="bold", fontsize=10)
                _clean_ax(ax)

        else:  # age — 한 행 = 모델, 한 열 = 나이 bin (해당 bin만 색칠 + 나머지 회색)
            bins = age_bins or []
            # 선택된 데이터셋 전부 합쳐서 좌표/나이 만든 후 bin별로 강조
            all_coords, all_ages = [], []
            for dname in selected_datasets:
                e = md.get(dname)
                if e is None or len(e) < 4 or e[3] is None:
                    continue
                all_coords.append(e[0]); all_ages.append(e[3])
            if not all_coords:
                # 빈 행 placeholder
                for c in range(n_cols):
                    ax = fig.add_subplot(gs[r, c])
                    ax.text(0.5, 0.5, "No age data", ha="center", va="center")
                    _clean_ax(ax)
                continue
            coords = np.concatenate(all_coords, axis=0)
            ages = np.concatenate(all_ages, axis=0)
            bidx = assign_age_bin_labels(ages, bins)

            for c, (lo, hi) in enumerate(bins):
                ax = fig.add_subplot(gs[r, c])
                target = bidx == c
                rest = ~target  # 다른 bin + invalid 모두 회색
                if rest.any():
                    ax.scatter(coords[rest, 0], coords[rest, 1], c="#DDDDDD",
                               s=max(1, point_size - 1), alpha=0.25,
                               rasterized=True, linewidths=0, zorder=1)
                color = AGE_PALETTE[c % len(AGE_PALETTE)]
                ax.scatter(coords[target, 0], coords[target, 1], c=color,
                           s=point_size, alpha=point_alpha,
                           rasterized=True, linewidths=0, zorder=2)
                hi_label = int(hi) if hi < 200 else "+"
                title = f"{int(lo)}–{hi_label} (n={int(target.sum()):,})"
                if r == 0:
                    ax.set_title(title, fontweight="bold", pad=6)
                else:
                    # 행마다 동일 title이라 첫 행에만 표시. 각 셀 우상단에 n만 작게.
                    ax.text(0.98, 0.97, f"n={int(target.sum()):,}",
                            transform=ax.transAxes, fontsize=7,
                            ha="right", va="top",
                            bbox=dict(facecolor="white", alpha=0.7,
                                      edgecolor="none", pad=2))
                if c == 0:
                    ax.set_ylabel(mname, fontweight="bold", fontsize=10)
                _clean_ax(ax)

    p = umap_params
    title = (f"{title_prefix}(n_neighbors={p['n_neighbors']}, "
             f"min_dist={p['min_dist']}, metric={p['metric']}, "
             f"L2={'on' if p.get('do_l2') else 'off'})")
    fig.suptitle(title, fontsize=12, y=1.01, fontweight="bold")
    # tight_layout는 suptitle/legend와 충돌 가능. GridSpec hspace/wspace로 충분.
    return fig, metrics


# ════════════════════════════════════════════════════════════════════════
# Ad-hoc 헬퍼 — 노트북 셀에서 한 줄로 호출하는 단순 함수들
# ════════════════════════════════════════════════════════════════════════
DEFAULT_EMB_DIR = "/home/irteam/local-node-d/tykim/visuallize/results/embeddings"

# 기본 데이터셋 (사용자 의견 — ptbxl + zzu 만)
DEFAULT_DATASETS = ("ptbxl", "zzu")

# 데이터셋 → "normal" 라벨 컬럼 (paper labels 기준).
# 여러 컬럼 OR 결합. 한 행이라도 1이면 normal.
NORMAL_LABEL_COLS = {
    "ptbxl": ["NORM"],
    "zzu":   ["Normal_ECG", "Otherwise_normal_ECG"],
}

# ─── ICD-10 진단 매핑 (cross-dataset 비교용) ─────────────────────────────────
# PTBXL all_paper_labels (약자) + ZZU bench_labels (is_*) → ICD prefix
# 사용자가 제공한 매핑을 csv 컬럼명에 맞게 변환.
ICD_LABEL_MAP: dict[str, dict[str, str]] = {
    "ptbxl": {
        # 사용자 매핑 (is_*) → PTBXL all_paper csv 컬럼 → ICD prefix
        "AFIB":  "I48",         # is_AF
        "AFLT":  "I48",         # is_AFL
        "SR":    "Z00.00",      # is_NSR
        "STACH": "R00.0",       # is_STach
        "SBRAD": "R00.1",       # is_SB / is_Brady
        "SARRH": "R00.8",       # is_SA
        "1AVB":  "I44.0",       # is_IAVB
        "CLBBB": "I44.7",       # is_CLBBB / is_LBBB
        "CRBBB": "I45.1",       # is_CRBBB / is_RBBB
        "IRBBB": "I45.1",       # is_IRBBB
        "LAFB":  "I44.4",       # is_LAnFB
        "LPFB":  "I44.5",       # left posterior fascicular block
        "LPR":   "I44.0",       # is_LPR (1AVB-equivalent)
        "IVCD":  "I45.4",       # is_NSIVCB
        "LNGQT": "I45.81",      # is_LQT
        "PAC":   "I49.1",       # is_PAC / is_SVPB
        "PVC":   "I49.3",       # is_PVC / is_VPB
        "PSVT":  "I47.1",       # supraventricular tachy
        "WPW":   "I45.6",       # is_WPW (subclass)
        "QWAVE": "R94.31",      # is_QAb
        "INVT":  "R94.31",      # is_TInv
        "TAB_":  "R94.31",      # is_TAb
        "LVH":   "I51.7",       # left ventricular hypertrophy
        "RVH":   "I51.7",
        "LAO_LAE": "I51.7",     # left atrial enlargement
        "RAO_RAE": "I51.7",
        "LVOLT": "R94.31",      # is_LQRSV
        "PRCS":  "R94.31",      # is_PRWP (poor R-wave progression)
        "LAD":   "R94.31",      # is_LAD
        "PACE":  "Z95.0",       # is_PR (pacemaker)
        "NORM":  "Z00.00",
    },
    "zzu": {
        # ZZU bench labels (is_*) — 사용자 매핑 그대로
        "is_STach":         "R00.0",
        "is_SBrady":        "R00.1",
        "is_SArr":          "R00.8",
        "is_RBBB":          "I45.1",
        "is_IRBBB":         "I45.1",
        "is_LAnFB":         "I44.4",
        "is_LQT":           "I45.81",
        "is_LQTc":          "I45.81",
        "is_JTach":         "I47.1",
        "is_JEsc":          "I49.8",
        "is_EATach":        "I47.1",
        "is_IAVB":          "I44.0",
        "is_CAVB":          "I44.2",
        "is_WPW":           "I45.6",
        "is_AVDiss":        "I45.89",
        "is_AEsc":          "I49.8",
        "is_RVH":           "I51.7",
        "is_LVH":           "I51.7",
        "is_LVHV":          "R94.31",
        "is_RAE":           "I51.7",
        "is_LAE":           "I51.7",
        "is_Hyperkalemia":  "E87.5",
        "is_Hypocalcemia":  "E83.51",
        "is_Hypokalemia":   "E87.6",
        "is_Normal":        "Z00.00",
        "is_Normal_Other":  "Z00.00",
    },
}

# ICD prefix → 사람-가독 표시명 (figure legend용)
ICD_DISPLAY = {
    "Z00.00":  "Normal/SR",
    "I48":     "AFib/AFlut",
    "R00.0":   "S. Tachy",
    "R00.1":   "S. Brady",
    "R00.8":   "S. Arr",
    "I44.0":   "1° AVB",
    "I44.2":   "Complete AVB",
    "I44.4":   "LAFB",
    "I44.5":   "LPFB",
    "I44.7":   "LBBB",
    "I45.1":   "RBBB",
    "I45.4":   "Other IVCD",
    "I45.6":   "WPW",
    "I45.81":  "Long QT",
    "I45.89":  "AV Dissoc",
    "I47.1":   "SVT",
    "I49.1":   "PAC/SVPB",
    "I49.3":   "PVC",
    "I49.8":   "Junct/AEsc",
    "I51.7":   "Hypertrophy/Enlarge",
    "E87.5":   "Hyperkalemia",
    "E87.6":   "Hypokalemia",
    "E83.51":  "Hypocalcemia",
    "Z95.0":   "Paced",
    "R94.31":  "Other ab. ECG",
}

# ZZU는 paper 외에 bench csv도 (is_* 매핑용)
LABEL_CSV_BENCH_HINTS = {
    "ptbxl": "ptbxl_all_paper_labels.csv",
    "zzu":   "zzu_bench_labels.csv",
}

# ─── PTBXL super class 매핑 (NORM/MI/STTC/CD/HYP) ───────────────────────────
SUPER_DISPLAY = {
    "NORM": "Normal",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD":   "Conduction Disturbance",
    "HYP":  "Hypertrophy",
}

SUPER_CSV_HINTS = {
    "ptbxl": "ptbxl_super_paper_labels.csv",   # NORM/MI/STTC/CD/HYP 컬럼 직접
    "zzu":   "zzu_paper_labels.csv",           # paper labels (MI 포함)
}

# ZZU paper labels → PTBXL super class
SUPER_LABEL_MAP = {
    "ptbxl": {  # 컬럼 그대로 super 코드
        "NORM": "NORM", "MI": "MI", "STTC": "STTC", "CD": "CD", "HYP": "HYP",
    },
    "zzu": {
        # NORM
        "Normal_ECG":           "NORM",
        "Otherwise_normal_ECG": "NORM",
        # MI / 허혈
        "Anteroseptal_MI":      "MI",
        "Ischemia":             "MI",
        # STTC: ST/T 변화 + 재분극
        "T_wave_abnormality":         "STTC",
        "ST_deviation":               "STTC",
        "ST_deviation_with_T_wave_change": "STTC",
        "Prominent_U_waves":          "STTC",
        "Early_repolarization":       "STTC",
        "TU_fusion":                  "STTC",
        "Prolonged_QT_interval":      "STTC",
        "Short_QT_interval":          "STTC",
        # CD: AV/intraventricular 전도장애 + arrhythmia
        "2_1_AV_block":                                  "CD",
        "AV_block_advanced_high_grade":                  "CD",
        "AV_block_complete_third_degree":                "CD",
        "AV_block_varying_conduction":                   "CD",
        "AV_dissociation":                               "CD",
        "Second_degree_AV_block_Mobitz_type_IWenckebach":"CD",
        "Second_degree_AV_block_Mobitz_type_II":         "CD",
        "Prolonged_PR_interval":                         "CD",
        "Short_PR_interva":                              "CD",  # csv 컬럼 오타 그대로
        "Atrial_fibrillation":                           "CD",
        "Atrial_flutter":                                "CD",
        "Atrial_premature_complexes_nonconducted":       "CD",
        "Ectopic_atrial_tachycardia_unifocal":           "CD",
        "Junctional_escape_complexes":                   "CD",
        "Junctional_premature_complexes":                "CD",
        "Junctional_tachycardia":                        "CD",
        "Left_anterior_fascicular_block":                "CD",
        "Left_bundle_branch_block":                      "CD",
        "Left_posterior_fascicular_block":               "CD",
        "Right_bundle_branch_block":                     "CD",
        "Incomplete_right_bundle_branch_block":          "CD",
        "Intraventricular_conduction_delay":             "CD",
        "Sinus_pause_or_arrest":                         "CD",
        "Supraventricular_tachycardia":                  "CD",
        "Ventricular_escape_complexes":                  "CD",
        "Ventricular_preexcitation":                     "CD",
        "Ventricular_tachycardia":                       "CD",
        # HYP: 비대/확장
        "Left_atrial_enlargement":      "HYP",
        "Left_ventricular_hypertrophy": "HYP",
        "Right_atrial_enlargement":     "HYP",
        "Right_ventricular_hypertrophy":"HYP",
        # 매핑 안 함 (mapping out of super):
        # Brugada_abnormality, Electrical_alternans, Fusion_complexes,
        # Hyper/Hypo K/Ca, Left/Right_axis_deviation, Low_voltage,
        # Ostium_primum_ASD, Sinus_arrhythmia, Sinus_bradycardia,
        # Sinus_tachycardia, AV_conduction_ratio_N_D
    },
}


@lru_cache(maxsize=8)
def _load_super_codes_aligned(dataset: str, n_rows: int):
    """SUPER_LABEL_MAP 기반 row별 대표 super class (NORM 우선)."""
    if dataset not in SUPER_LABEL_MAP:
        return None, []
    lmap = SUPER_LABEL_MAP[dataset]
    csv_name = SUPER_CSV_HINTS.get(dataset)
    csv_path = _resolve_label_csv(csv_name) if csv_name else None
    table_csv = DATASET_TABLE_HINTS.get(dataset)
    if csv_path is None or not table_csv or not Path(table_csv).exists():
        return None, []
    try:
        ldf = pd.read_csv(csv_path, low_memory=False)
        tdf = pd.read_csv(table_csv, low_memory=False, usecols=lambda c: c == "filepath")
    except Exception:
        return None, []
    if "filepath" not in ldf.columns or "filepath" not in tdf.columns:
        return None, []
    label_cols = [c for c in lmap if c in ldf.columns]
    if not label_cols:
        return None, []
    merged = tdf[["filepath"]].merge(
        ldf[["filepath"] + label_cols].drop_duplicates(subset="filepath"),
        on="filepath", how="left",
    ).iloc[:n_rows]
    mat = merged[label_cols].to_numpy(dtype=np.float32, na_value=0.0) > 0

    norm_cols = [c for c in label_cols if lmap[c] == "NORM"]
    other_cols = [c for c in label_cols if lmap[c] != "NORM"]
    name_to_idx = {c: i for i, c in enumerate(label_cols)}
    out = []
    for r in mat:
        chosen = None
        for c in norm_cols:
            if r[name_to_idx[c]]:
                chosen = "NORM"; break
        if chosen is None:
            for c in other_cols:
                if r[name_to_idx[c]]:
                    chosen = lmap[c]; break
        out.append(chosen)
    uniq = sorted({c for c in out if c is not None},
                  key=lambda x: list(SUPER_DISPLAY.keys()).index(x)
                                if x in SUPER_DISPLAY else 999)
    return out, uniq


def get_super_codes(dataset: str, n_rows: int):
    """샘플별 PTBXL super class 코드."""
    return _load_super_codes_aligned(dataset, n_rows)


@lru_cache(maxsize=8)
def _load_icd_codes_aligned(dataset: str, n_rows: int) -> tuple[Optional[list], list]:
    """
    ICD_LABEL_MAP 의 컬럼들을 데이터셋의 적합 csv (paper/bench)에서 읽어
    각 row 의 *대표 ICD 코드 1개* 를 정한다. 정상이 우선, 그 다음 발견 순서.
    return: (per_row_code list[N] (None=라벨 없음/미매핑), unique_codes list)
    """
    if dataset not in ICD_LABEL_MAP:
        return None, []
    code_map = ICD_LABEL_MAP[dataset]
    csv_name = LABEL_CSV_BENCH_HINTS.get(dataset)
    csv_path = _resolve_label_csv(csv_name) if csv_name else None
    table_csv = DATASET_TABLE_HINTS.get(dataset)
    if csv_path is None or not table_csv or not Path(table_csv).exists():
        return None, []
    try:
        ldf = pd.read_csv(csv_path, low_memory=False)
        tdf = pd.read_csv(table_csv, low_memory=False, usecols=lambda c: c == "filepath")
    except Exception:
        return None, []
    if "filepath" not in ldf.columns or "filepath" not in tdf.columns:
        return None, []
    # 매핑 가능한 컬럼만
    label_cols = [c for c in code_map if c in ldf.columns]
    if not label_cols:
        return None, []
    merged = tdf[["filepath"]].merge(
        ldf[["filepath"] + label_cols].drop_duplicates(subset="filepath"),
        on="filepath", how="left",
    ).iloc[:n_rows]
    mat = merged[label_cols].to_numpy(dtype=np.float32, na_value=0.0) > 0
    # row별 첫 활성 라벨의 ICD 코드 (Normal 우선)
    normal_cols = [c for c in label_cols if code_map[c] == "Z00.00"]
    other_cols = [c for c in label_cols if code_map[c] != "Z00.00"]
    out: list[Optional[str]] = []
    name_to_idx = {c: i for i, c in enumerate(label_cols)}
    for r in mat:
        # Normal 우선
        chosen = None
        for c in normal_cols:
            if r[name_to_idx[c]]:
                chosen = code_map[c]; break
        if chosen is None:
            for c in other_cols:
                if r[name_to_idx[c]]:
                    chosen = code_map[c]; break
        out.append(chosen)
    uniq = sorted({c for c in out if c is not None},
                  key=lambda x: list(ICD_DISPLAY.keys()).index(x)
                                if x in ICD_DISPLAY else 999)
    return out, uniq


def get_dx_codes(dataset: str, n_rows: int):
    """샘플별 ICD 코드 (대표 1개) 와 유니크 목록."""
    return _load_icd_codes_aligned(dataset, n_rows)


def dx_compatibility_table(datasets=DEFAULT_DATASETS,
                            sizes: Optional[dict] = None) -> pd.DataFrame:
    """
    데이터셋(default ptbxl+zzu) 간 ICD 진단 호환성 표.

    Returns DataFrame:
        icd, name, {ds}_mapped, {ds}_n, compatible (모든 ds에 sample ≥1)
    """
    from collections import Counter
    if isinstance(datasets, str):
        datasets = [datasets]
    if sizes is None:
        sizes = {"ptbxl": 21836, "zzu": 12327, "chapman": 10232,
                 "sph": 25770, "code15": 30000, "mimic4": 30000}

    counts: dict[str, Counter] = {}
    mapped_keys: dict[str, set] = {}
    for d in datasets:
        n = sizes.get(d, 30000)
        codes, _ = get_dx_codes(d, n)
        cnt = Counter(c for c in (codes or []) if c)
        counts[d] = cnt
        mapped_keys[d] = set(ICD_LABEL_MAP.get(d, {}).values())

    all_codes = set()
    for d in datasets:
        all_codes |= mapped_keys[d]
    all_codes = sorted(all_codes,
                       key=lambda x: (list(ICD_DISPLAY).index(x)
                                      if x in ICD_DISPLAY else 999, x))
    rows = []
    for code in all_codes:
        row = {"icd": code, "name": ICD_DISPLAY.get(code, code)}
        for d in datasets:
            n = counts[d].get(code, 0)
            row[f"{d}_n"] = n
            total = sizes.get(d, 30000)
            row[f"{d}_pct"] = round(100 * n / total, 2) if total else 0.0
        row["compatible"] = all(counts[d].get(code, 0) > 0 for d in datasets)
        rows.append(row)
    return pd.DataFrame(rows)


def is_normal_mask(dataset: str, n_rows: int) -> Optional[np.ndarray]:
    """paper labels에서 normal 여부 (bool, length n_rows). 매핑 없으면 None."""
    cols_for = NORMAL_LABEL_COLS.get(dataset)
    if not cols_for:
        return None
    mat, all_cols = get_labels(dataset)
    if mat is None:
        return None
    idxs = [all_cols.index(c) for c in cols_for if c in all_cols]
    if not idxs:
        return None
    sub = mat[:n_rows, idxs] > 0
    return sub.any(axis=1)


def get_emb(model: str, dataset: str, emb_dir: str | Path = DEFAULT_EMB_DIR):
    """`{model}_{dataset}.npy` 임베딩을 반환. 모델명은 표시명 (예: 'MERL (ResNet)')."""
    safe = sanitize(model)
    return np.load(Path(emb_dir) / f"{safe}_{dataset}.npy")


def get_coords(model: str, datasets, preset: str = "orig",
               emb_dir: str | Path = DEFAULT_EMB_DIR):
    """
    캐시된 UMAP 좌표를 반환. 데이터셋이 여러 개면 알파벳 정렬 후 concat 좌표.
    캐시가 없으면 즉석 계산해서 저장.

    return: (coords (N, 2), sizes dict {ds: n})
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    ordered = sorted(datasets)
    safe = sanitize(model)
    emb_dir = Path(emb_dir)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset} (available: {list(PRESETS)})")
    p = PRESETS[preset]

    # 1) 임베딩 로드 (size 계산용)
    emb_by_ds = {d: np.load(emb_dir / f"{safe}_{d}.npy") for d in ordered}
    sizes = {d: len(emb_by_ds[d]) for d in ordered}

    # 2) 캐시 시도
    load_cands, save_fn = cache_filenames_for(safe, p["tag"], tuple(ordered))
    cached, _ = try_load_cached_combined(emb_dir, ordered, sizes, load_cands)
    if cached is not None:
        return np.concatenate([cached[d] for d in ordered], axis=0), sizes

    # 3) 새로 계산 + 저장
    coords_by_ds, _ = compute_combined_for_model(
        emb_dir, safe, ordered, emb_by_ds, p, seed=42,
        try_cache=False, save_after=True,
    )
    return np.concatenate([coords_by_ds[d] for d in ordered], axis=0), sizes


def get_labels(dataset: str) -> tuple[Optional[np.ndarray], list[str]]:
    """paper label 매트릭스 (N, K) + label_cols 반환. 없으면 (None, [])."""
    # 임베딩 행수 (모든 모델 동일) 추정 — table_csv 행수 사용
    table = DATASET_TABLE_HINTS.get(dataset)
    if not table or not Path(table).exists():
        return None, []
    n_rows = sum(1 for _ in open(table)) - 1  # header 제외
    return _load_paper_labels_aligned(dataset, n_rows)


def get_ages(dataset: str) -> Optional[np.ndarray]:
    """table_csv의 age 컬럼 (scaled) 을 반환. 없으면 None."""
    return _load_age_cached(dataset, json.dumps({}))


def ds_split(coords: np.ndarray, sizes: dict[str, int]) -> dict[str, np.ndarray]:
    """get_coords 가 반환한 concat 좌표를 데이터셋별로 다시 split."""
    out, off = {}, 0
    for d in sorted(sizes.keys()):
        n = sizes[d]
        out[d] = coords[off:off + n]
        off += n
    return out


def quick(model: str, datasets=DEFAULT_DATASETS, preset: str = "orig",
          color: str = "dataset", label: Optional[str] = None,
          age_bins=(0, 18, 40, 60, 80, 200),
          emb_dir: str | Path = DEFAULT_EMB_DIR,
          ax=None, s: int = 3, alpha: float = 0.5,
          show: bool = True):
    """
    한 줄로 UMAP을 그린다. matplotlib axes에 점만 찍는 ad-hoc 함수.

    예:
        quick("CPC", "ptbxl")
        quick("CPC", ["ptbxl","zzu"])
        quick("CPC", "ptbxl", color="age")
        quick("CPC", ["ptbxl","zzu"], color="age", age_bins=[0,40,60,200])
        quick("CPC", "ptbxl", color="label", label="NORM")
        quick("CPC", "ptbxl", color="label", label=("ptbxl","NORM"))  # 둘다 OK
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    ordered = sorted(datasets)
    coords, sizes = get_coords(model, ordered, preset=preset, emb_dir=emb_dir)
    coords_by_ds = ds_split(coords, sizes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=110)
    else:
        fig = ax.figure

    if color == "dataset":
        for ci, d in enumerate(ordered):
            c = coords_by_ds[d]
            ax.scatter(c[:, 0], c[:, 1],
                       c=PALETTE[ci % len(PALETTE)],
                       marker=MARKERS[ci % len(MARKERS)],
                       s=s, alpha=alpha, linewidths=0,
                       label=f"{d} (n={len(c):,})")
        ax.legend(fontsize=8, markerscale=2)

    elif color == "age":
        bins = parse_age_bins(age_bins)
        ages_parts = []
        for d in ordered:
            a = get_ages(d)
            n = sizes[d]
            if a is None:
                ages_parts.append(np.full(n, np.nan))
            else:
                ages_parts.append(a[:n] if len(a) >= n
                                  else np.concatenate([a, np.full(n - len(a), np.nan)]))
        ages = np.concatenate(ages_parts)
        bidx = assign_age_bin_labels(ages, bins)
        # 무효 회색
        inv = bidx < 0
        if inv.any():
            ax.scatter(coords[inv, 0], coords[inv, 1], c="#DDDDDD",
                       s=max(1, s - 1), alpha=0.25, linewidths=0)
        for bi, (lo, hi) in enumerate(bins):
            m = bidx == bi
            if not m.any():
                continue
            color_i = AGE_PALETTE[bi % len(AGE_PALETTE)]
            ax.scatter(coords[m, 0], coords[m, 1], c=color_i,
                       s=s, alpha=alpha, linewidths=0,
                       label=f"{int(lo)}–{int(hi) if hi<200 else '+'} (n={int(m.sum()):,})")
        ax.legend(fontsize=8, markerscale=2, loc="best")

    elif color == "label":
        # label은 단일 데이터셋에서만 의미
        if len(ordered) > 1:
            raise ValueError("color='label'은 단일 데이터셋에서만 지원")
        d = ordered[0]
        # label 인자 정규화
        if isinstance(label, tuple):
            ds_n, lbl_n = label
            if ds_n != d:
                raise ValueError(f"label dataset '{ds_n}' != requested '{d}'")
        else:
            lbl_n = label
        if lbl_n is None:
            raise ValueError("color='label'은 label='이름' 인자가 필요")
        mat, cols = get_labels(d)
        if mat is None or lbl_n not in cols:
            raise ValueError(f"{d}에 라벨 '{lbl_n}' 없음. 가능: {cols[:8]}...")
        li = cols.index(lbl_n)
        n = sizes[d]
        bin_ = (mat[:n, li] > 0).astype(int)
        pmask = bin_ == 1
        ax.scatter(coords[~pmask, 0], coords[~pmask, 1], c="#e74c3c",
                   s=s, alpha=alpha, linewidths=0,
                   label=f"neg (n={int((~pmask).sum()):,})")
        ax.scatter(coords[pmask, 0], coords[pmask, 1], c="#2ecc71",
                   s=s, alpha=alpha, linewidths=0,
                   label=f"{lbl_n} pos (n={int(pmask.sum()):,})")
        ax.legend(fontsize=8, markerscale=2)

    else:
        raise ValueError(f"color='{color}' (dataset|age|label 중 하나)")

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(
        f"{model} · {'+'.join(ordered)} · {preset} · color={color}"
        + (f" ({label})" if color == "label" else ""),
        fontsize=10,
    )
    if show:
        plt.show()
    return ax


def quick_dx(models="all", datasets=DEFAULT_DATASETS,
             top_n: int = 10,
             include_codes: Optional[Iterable[str]] = None,
             exclude_codes: Optional[Iterable[str]] = None,
             age_split: Optional[float] = 18.0,
             balance_per_code: Optional[int] = None,
             balance_per_group_code: Optional[int] = None,
             balance_strict_groups: bool = False,
             palette: Optional[list] = None,
             code_scheme: str = "icd",
             preset: str = "orig",
             emb_dir: str | Path = DEFAULT_EMB_DIR,
             s: int = 3, alpha: float = 0.6,
             figsize_per_cell: tuple = (3.4, 3.0),
             show_metrics: bool = True,
             seed: int = 42,
             coord_clip_pct: Optional[float] = 99.5,
             show: bool = True):
    """
    각 모델의 UMAP을 ICD-10 진단 코드별로 색칠한다.
    age_split이 주어지면 (combined / adult ≥ split / pediatric < split) 3-col 비교.

    layout (age_split=None): rows = models, cols = (UMAP, metrics)
    layout (age_split given): rows = models, cols = (combined, adult, pediatric, metrics?)
        — 각 axes에서 해당 그룹만 진단색, 나머지(다른 그룹+미매핑)는 회색.

    Args
    ----
    models: 'all' (default 4개) 또는 list[str]
    datasets: ('ptbxl', 'zzu') default
    top_n: 데이터셋 합산 빈도 기준 상위 N개 ICD 코드 (legend 가독성)
    include_codes: 명시적 ICD 코드 리스트 (top_n 무시)
    age_split: 성인/소아 분리 기준 (default 18). None이면 전체만.
    """
    from collections import Counter
    if models == "all" or models is None:
        models = list_models(emb_dir)
    elif isinstance(models, str):
        models = [models]
    if isinstance(datasets, str):
        datasets = [datasets]
    ordered = sorted(datasets)

    # 1) code scheme 별 디스패처
    if code_scheme == "super":
        _code_loader = _load_super_codes_aligned
        _display_map = SUPER_DISPLAY
    else:  # default 'icd'
        _code_loader = _load_icd_codes_aligned
        _display_map = ICD_DISPLAY

    # 데이터셋 합산해서 가장 흔한 코드 결정
    code_counter: Counter = Counter()
    per_ds_codes: dict[str, list[Optional[str]]] = {}
    for d in ordered:
        any_emb = get_emb(models[0], d, emb_dir=emb_dir)
        n = len(any_emb)
        codes, _ = _code_loader(d, n)
        if codes is None:
            codes = [None] * n
        per_ds_codes[d] = codes
        code_counter.update([c for c in codes if c is not None])
    keep_codes = (list(include_codes) if include_codes is not None
                  else [c for c, _ in code_counter.most_common(top_n)])
    if exclude_codes:
        excl = set(exclude_codes)
        keep_codes = [c for c in keep_codes if c not in excl]
    pal = palette if palette is not None else DX_PALETTE
    code_to_color = {c: pal[i % len(pal)]
                     for i, c in enumerate(keep_codes)}
    keep_set = set(keep_codes)

    # 2) age 그룹 정의
    if age_split is None:
        age_groups = [("combined", lambda a: np.ones(len(a), dtype=bool))]
    else:
        age_groups = [
            ("combined",  lambda a: np.ones(len(a), dtype=bool)),
            (f"adult ≥{age_split:g}",     lambda a: np.isfinite(a) & (a >= age_split)),
            (f"pediatric <{age_split:g}", lambda a: np.isfinite(a) & (a < age_split)),
        ]
    n_age_cols = len(age_groups)

    # 3) layout
    n_rows = len(models)
    n_cols = n_age_cols + (1 if show_metrics else 0)
    width_ratios = [figsize_per_cell[0]] * n_age_cols + (
        [2.6] if show_metrics else [])
    fig_w = sum(width_ratios) + 0.8
    legend_h = 1.5
    fig_h = figsize_per_cell[1] * n_rows + 0.8 + legend_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=110)
    gs = fig.add_gridspec(
        n_rows, n_cols,
        width_ratios=width_ratios,
        hspace=0.16, wspace=0.05,
        left=0.06, right=0.99,
        top=1 - (legend_h + 0.4) / fig_h, bottom=0.02,
    )

    metrics_summary: dict[str, dict] = {}
    global_legend_handles: list = []

    for r, m in enumerate(models):
        coords, sizes = get_coords(m, ordered, preset=preset, emb_dir=emb_dir)
        # axes-limit clip (per-model): UMAP 좌표상 outlier 1–2개가 axes 를
        #모두 잡아먹는 경우를 위해, 모델 전체 좌표의 percentile 로 zoom.
        # coord_clip_pct=99.5 → 양 끝 0.5% 는 화면 밖. 점 자체는 그대로 그리되
        # axes 는 dense main blob 에 fit 됨.
        if coord_clip_pct is not None and len(coords) > 0:
            lo = (100.0 - coord_clip_pct) / 2.0
            hi = 100.0 - lo
            xlo, xhi = np.percentile(coords[:, 0], [lo, hi])
            ylo, yhi = np.percentile(coords[:, 1], [lo, hi])
            xpad = (xhi - xlo) * 0.04 if xhi > xlo else 1.0
            ypad = (yhi - ylo) * 0.04 if yhi > ylo else 1.0
            clip_xlim = (xlo - xpad, xhi + xpad)
            clip_ylim = (ylo - ypad, yhi + ypad)
        else:
            clip_xlim = clip_ylim = None
        # codes_concat
        codes_concat: list[Optional[str]] = []
        for d in ordered:
            n = sizes[d]
            cs = per_ds_codes[d]
            if len(cs) >= n:
                codes_concat.extend(cs[:n])
            else:
                codes_concat.extend(cs + [None] * (n - len(cs)))
        # ages_concat
        ages_concat = []
        for d in ordered:
            n = sizes[d]
            a = get_ages(d)
            if a is None:
                ages_concat.append(np.full(n, np.nan))
            else:
                ages_concat.append(a[:n] if len(a) >= n
                                   else np.concatenate([a, np.full(n - len(a), np.nan)]))
        ages_concat = np.concatenate(ages_concat)

        # 진단 매핑 array (코드 외/None은 모두 회색)
        is_in_keep = np.array([c in keep_set for c in codes_concat])

        metrics_summary[m] = {}

        rng = np.random.RandomState(seed)

        # balance_strict_groups: 모든 (그룹 × 진단) cell 중 *최소값* 으로 동일 cap
        # → 진단별 균등 + 성인/소아 균등 (전체 cell 같은 sample 수)
        cap_strict_per_code: dict[str, int] = {}
        if balance_strict_groups and len(age_groups) > 1:
            grp_only = [(gn, mf) for gn, mf in age_groups if 'combined' not in gn]
            codes_arr = np.asarray(codes_concat, dtype=object)
            all_cell_sizes = []
            for code in keep_codes:
                code_mask = codes_arr == code
                for _, mf in grp_only:
                    n = int((mf(ages_concat) & code_mask).sum())
                    if n > 0:
                        all_cell_sizes.append(n)
            cap_uniform = min(all_cell_sizes) if all_cell_sizes else 0
            cap_strict_per_code = {code: cap_uniform for code in keep_codes}

        for c, (gname, mask_fn) in enumerate(age_groups):
            ax = fig.add_subplot(gs[r, c])
            grp = mask_fn(ages_concat)

            # cap 우선순위:
            #   strict_groups (자동 min) > balance_per_group_code > balance_per_code
            keep_idx_per_code: dict[str, np.ndarray] = {}
            for code in keep_codes:
                cand = np.where(grp & np.array([cc == code for cc in codes_concat]))[0]
                if balance_strict_groups:
                    cap_code = cap_strict_per_code.get(code, len(cand))
                elif balance_per_group_code is not None:
                    cap_code = balance_per_group_code
                else:
                    cap_code = balance_per_code
                if cap_code is not None and len(cand) > cap_code:
                    cand = rng.choice(cand, cap_code, replace=False)
                keep_idx_per_code[code] = cand
            _cap = (cap_strict_per_code  # display title 용
                    if balance_strict_groups else
                    (balance_per_group_code or balance_per_code))
            # 그룹 내 미매핑(None/제외)
            unmapped_idx = np.where(grp & ~is_in_keep)[0]
            # 다른 그룹 (회색 배경)
            outside_idx = np.where(~grp)[0]

            # balance 시 회색도 동일 cap 으로 cap (시각 균형)
            # 회색 cap — strict 모드에선 진단 cap 의 max 기준
            if balance_strict_groups and cap_strict_per_code:
                grey_cap_val = max(cap_strict_per_code.values())
            elif isinstance(_cap, int):
                grey_cap_val = _cap
            else:
                grey_cap_val = None
            if grey_cap_val is not None:
                if len(unmapped_idx) > grey_cap_val:
                    unmapped_idx = rng.choice(unmapped_idx, grey_cap_val, replace=False)
                cap_outside = grey_cap_val * max(2, len(keep_codes))
                if len(outside_idx) > cap_outside:
                    outside_idx = rng.choice(outside_idx, cap_outside, replace=False)

            # 회색 그리기 (다른 그룹 + 그룹 내 미매핑)
            grey_idx = np.concatenate([outside_idx, unmapped_idx])
            if len(grey_idx) > 0:
                ax.scatter(coords[grey_idx, 0], coords[grey_idx, 1], c="#DDDDDD",
                           s=max(1, s - 1), alpha=0.18, linewidths=0, zorder=1)

            # 코드별 색칠
            cell_handles = []
            for i, code in enumerate(keep_codes):
                sel = keep_idx_per_code[code]
                if len(sel) == 0:
                    continue
                color = code_to_color[code]
                ax.scatter(coords[sel, 0], coords[sel, 1], c=color,
                           s=s, alpha=alpha, linewidths=0, zorder=2 + i)
                disp = _display_map.get(code, code)
                cell_handles.append(mpatches.Patch(
                    facecolor=color, label=f"{disp} ({code})"
                ))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.patch.set_facecolor("#FAFAFA")
            if clip_xlim is not None:
                ax.set_xlim(*clip_xlim); ax.set_ylim(*clip_ylim)

            # title (첫 행만) — 길면 axes 폭 넘으니 짧게 + 두 줄
            n_drawn = int(sum(len(v) for v in keep_idx_per_code.values()))
            if r == 0:
                ax.set_title(f"{gname}\nn={int(grp.sum()):,}",
                             fontweight="bold", fontsize=10, pad=4)
            if c == 0:
                ax.set_ylabel(m, fontweight="bold", fontsize=11,
                              rotation=0, labelpad=8, ha="right", va="center")

            # legend handles는 첫 row + combined col에서만 모음
            if r == 0 and c == 0:
                global_legend_handles = cell_handles

            # 그룹별 metrics — balance 적용된 인덱스 기준 (more reliable when imbalanced)
            metric_idx = np.concatenate(
                [keep_idx_per_code[code] for code in keep_codes
                 if len(keep_idx_per_code[code]) > 0]
            ) if any(len(v) for v in keep_idx_per_code.values()) else np.array([], dtype=int)
            if len(metric_idx) > 0:
                sub_codes = [codes_concat[i] for i in metric_idx]
                sub_coords = coords[metric_idx]
                metrics_summary[m][gname] = _compute_dx_metrics(
                    sub_coords, sub_codes, keep_codes)
            else:
                metrics_summary[m][gname] = {}

        # metrics 표 col (combined 그룹 metrics)
        if show_metrics:
            ax_t = fig.add_subplot(gs[r, n_age_cols])
            ax_t.axis("off")
            _render_metrics_table(
                ax_t, metrics_summary[m].get("combined", {}),
                title=("metrics (combined)" if r == 0 else None),
                display_map=_display_map,
            )

    # 통합 legend (figure 상단)
    if global_legend_handles:
        ncol = min(len(global_legend_handles), 4)
        fig.legend(
            handles=global_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1 - 0.45 / fig_h),
            fontsize=9, ncol=ncol, frameon=False,
            handlelength=1.2, handletextpad=0.5, columnspacing=1.2,
        )
    age_str = (f" · age split={age_split:g}" if age_split is not None else "")
    if balance_strict_groups:
        # cap_uniform 값 표기
        if isinstance(_cap, dict) and _cap:
            uniform_n = next(iter(_cap.values()))
            bal_str = f" · strict-balance (uniform n={uniform_n}/cell)"
        else:
            bal_str = " · strict-balance"
    elif isinstance(_cap, int):
        bal_str = f" · balanced≤{_cap}/code"
    else:
        bal_str = ""
    scheme_str = "super" if code_scheme == "super" else "ICD-10"
    fig.suptitle(
        f"{len(models)} models · {'+'.join(ordered)} · "
        f"diagnosis ({scheme_str}){age_str}{bal_str} · {preset}",
        fontsize=12, fontweight="bold",
        y=1 - 0.15 / fig_h,
    )
    if show:
        plt.show()
    return fig, metrics_summary


def _compute_dx_metrics(coords: np.ndarray, codes_per_row,
                        keep_codes) -> dict:
    """
    코드별 silhouette (one-vs-rest) + kNN-BACC (one-vs-rest).
    소수 클래스 (<10 sample) 는 nan 처리.
    """
    codes_arr = np.asarray(codes_per_row, dtype=object)
    out = {}
    for code in keep_codes:
        mask = codes_arr == code
        n_pos = int(mask.sum())
        if n_pos < 10 or n_pos > len(mask) - 10:
            out[code] = {"sil": float("nan"), "bacc": float("nan"), "n_pos": n_pos}
            continue
        binary = mask.astype(int)
        sil, bacc = compute_metrics(coords, binary)
        out[code] = {"sil": sil, "bacc": bacc, "n_pos": n_pos}
    return out


def _render_metrics_table(ax, metrics: dict, title=None, display_map=None):
    """ax 에 텍스트 테이블로 코드별 sil / bacc 표시."""
    if display_map is None:
        display_map = ICD_DISPLAY
    rows = [["code", "sil", "kNN-BACC", "n"]]
    for code, m in metrics.items():
        sil = m["sil"]; bacc = m["bacc"]
        sil_s = f"{sil:.3f}" if np.isfinite(sil) else "—"
        bacc_s = f"{bacc:.3f}" if np.isfinite(bacc) else "—"
        rows.append([
            display_map.get(code, code)[:14],
            sil_s, bacc_s, f"{m['n_pos']:,}"
        ])
    table = ax.table(cellText=rows, loc="center", cellLoc="left",
                     colWidths=[0.34, 0.18, 0.24, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.18)
    # 헤더 굵게
    for j in range(4):
        c = table[0, j]
        c.set_facecolor("#EEEEEE")
        c.set_text_props(weight="bold")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)


def quick_dx_raw(model: str, dataset: str = "ptbxl",
                 label_columns: Optional[Iterable[str]] = None,
                 top_n: int = 7,
                 label_csv: Optional[str] = None,
                 max_per_label: Optional[int] = None,
                 preset: str = "orig",
                 emb_dir: str | Path = DEFAULT_EMB_DIR,
                 ax=None, s: int = 4, alpha: float = 0.65,
                 figsize: tuple = (8, 6), seed: int = 42,
                 coord_clip_pct: Optional[float] = 99.5,
                 show: bool = True):
    """
    paper label CSV 의 컬럼명(약자) 그대로 색칠하는 단일 axes UMAP.
    여러 라벨이 동시 활성 시 행 순서상 처음 발견된 라벨 우선.

    예:
        # PTBXL diag paper 의 일부 약자
        quick_dx_raw('CPC', 'ptbxl',
                     label_columns=['AFIB','CRBBB','CLBBB','PVC','PAC','SR'])
        # chapman paper labels 그대로 (top_n 빈도순)
        quick_dx_raw('ECG-Founder', 'chapman', top_n=8)
    """
    from collections import Counter

    safe = sanitize(model)
    emb = np.load(Path(emb_dir) / f"{safe}_{dataset}.npy")
    coords, sizes = get_coords(model, [dataset], preset=preset, emb_dir=emb_dir)
    n_rows = sizes[dataset]

    # label CSV 결정
    if label_csv:
        csv_path = Path(label_csv)
    else:
        # ICD_LABEL_MAP 에 사용된 csv 우선, 없으면 paper csv
        csv_name = LABEL_CSV_BENCH_HINTS.get(dataset) or LABEL_CSV_HINTS.get(dataset)
        csv_path = _resolve_label_csv(csv_name) if csv_name else None
    if csv_path is None or not Path(csv_path).exists():
        raise ValueError(f"라벨 csv 없음: {csv_path}")

    table_csv = DATASET_TABLE_HINTS.get(dataset)
    if not table_csv:
        raise ValueError(f"table_csv 매핑 없음: {dataset}")

    ldf = pd.read_csv(csv_path, low_memory=False)
    tdf = pd.read_csv(table_csv, low_memory=False, usecols=lambda c: c == "filepath")
    all_label_cols = [c for c in ldf.columns if c not in _LABEL_SKIP_COLS]

    # 컬럼 선택
    if label_columns is None:
        # 전체 데이터셋 빈도 top_n
        merged_full = tdf[["filepath"]].merge(
            ldf[["filepath"] + all_label_cols].drop_duplicates(subset="filepath"),
            on="filepath", how="left"
        )
        full_mat = merged_full[all_label_cols].to_numpy(dtype=np.float32, na_value=0.0) > 0
        col_count = Counter()
        for j, c in enumerate(all_label_cols):
            col_count[c] = int(full_mat[:, j].sum())
        label_columns = [c for c, _ in col_count.most_common(top_n)]
    else:
        label_columns = [c for c in label_columns if c in all_label_cols]
    if not label_columns:
        raise ValueError("선택된 label_columns 없음")

    # filepath 정렬해 첫 n_rows
    merged = tdf[["filepath"]].merge(
        ldf[["filepath"] + label_columns].drop_duplicates(subset="filepath"),
        on="filepath", how="left",
    ).iloc[:n_rows]
    mat = merged[label_columns].to_numpy(dtype=np.float32, na_value=0.0) > 0

    # 각 sample의 대표 라벨 (첫 활성)
    rep = np.full(n_rows, -1, dtype=np.int32)
    for j in range(len(label_columns)):
        unset = rep < 0
        rep[unset & mat[:, j]] = j

    # 컬러 팔레트 — categorical-distinct (tab10 + tab20)
    colors = DX_PALETTE[:len(label_columns)]

    # 다운샘플 (max_per_label)
    rng = np.random.RandomState(seed)
    keep_idx_per_lbl: dict[int, np.ndarray] = {}
    for j in range(len(label_columns)):
        cand = np.where(rep == j)[0]
        if max_per_label and len(cand) > max_per_label:
            cand = rng.choice(cand, max_per_label, replace=False)
        keep_idx_per_lbl[j] = cand
    unmapped_idx = np.where(rep < 0)[0]
    if max_per_label and len(unmapped_idx) > max_per_label * 3:
        unmapped_idx = rng.choice(unmapped_idx, max_per_label * 3, replace=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=110)
    else:
        fig = ax.figure
    if len(unmapped_idx):
        ax.scatter(coords[unmapped_idx, 0], coords[unmapped_idx, 1],
                   c="#DDDDDD", s=max(1, s - 1), alpha=0.2,
                   linewidths=0, zorder=1)
    handles = []
    for j, col in enumerate(label_columns):
        sel = keep_idx_per_lbl[j]
        if not len(sel):
            continue
        ax.scatter(coords[sel, 0], coords[sel, 1], c=colors[j],
                   s=s, alpha=alpha, linewidths=0, zorder=2 + j,
                   label=f"{col} (n={len(sel):,})")
        handles.append(mpatches.Patch(facecolor=colors[j], label=col))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_facecolor("#FAFAFA")
    if coord_clip_pct is not None and len(coords) > 0:
        lo = (100.0 - coord_clip_pct) / 2.0
        hi = 100.0 - lo
        xlo, xhi = np.percentile(coords[:, 0], [lo, hi])
        ylo, yhi = np.percentile(coords[:, 1], [lo, hi])
        xpad = (xhi - xlo) * 0.04 if xhi > xlo else 1.0
        ypad = (yhi - ylo) * 0.04 if yhi > ylo else 1.0
        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)
    ax.set_title(f"{model} · {dataset} · "
                 f"raw paper labels ({len(label_columns)} cols)",
                 fontsize=10, fontweight="bold")
    # 한 줄 가로 legend (figure 상단)
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.03),
               ncol=min(len(handles), 8),
               fontsize=10, frameon=False,
               handlelength=0.8, handletextpad=0.4, columnspacing=1.5)
    if show:
        plt.show()
    return fig, label_columns


def list_models(emb_dir: str | Path = DEFAULT_EMB_DIR,
                exclude: Optional[Iterable[str]] = None) -> list[str]:
    """`emb_dir` 내 가용 모델 목록 (기본 정렬). default로 DEFAULT_EXCLUDED_MODELS 제외."""
    models, _, _ = discover_models_and_datasets(Path(emb_dir))
    excl = set(DEFAULT_EXCLUDED_MODELS if exclude is None else exclude)
    return [m for m in models if m not in excl]


def quick_age_na(models="all", datasets=DEFAULT_DATASETS,
                 age_bins=(0, 18, 40, 60, 80, 200),
                 preset: str = "orig",
                 emb_dir: str | Path = DEFAULT_EMB_DIR,
                 s: int = 3, alpha: float = 0.6,
                 figsize_per_cell: tuple = (2.4, 2.2), show: bool = True):
    """
    각 (모델 행 × 나이 bin 열) 셀:
      - 다른 나이대 점은 회색 배경
      - 해당 나이대만: normal=초록, abnormal=빨강

    예:
        quick_age_na()                                   # 모든 모델
        quick_age_na('CPC')                              # 단일 모델
        quick_age_na(['CPC', 'ECG-Founder'])             # 일부 모델
        quick_age_na(age_bins=[0, 30, 50, 70, 200])      # bin 변경
    """
    if models == "all" or models is None:
        models = list_models(emb_dir)
    elif isinstance(models, str):
        models = [models]
    if isinstance(datasets, str):
        datasets = [datasets]
    ordered = sorted(datasets)
    bins = parse_age_bins(age_bins)
    n_rows = len(models)
    n_cols = len(bins)
    if n_cols == 0:
        raise ValueError("age_bins가 비어 있음")

    fig_w = figsize_per_cell[0] * n_cols
    fig_h = figsize_per_cell[1] * n_rows + 0.6  # legend 영역
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        squeeze=False, dpi=110,
        gridspec_kw=dict(hspace=0.12, wspace=0.04,
                          left=0.06, right=0.995,
                          top=1 - 0.6 / fig_h, bottom=0.02),
    )

    pos_c, neg_c = "#2ecc71", "#e74c3c"

    for r, m in enumerate(models):
        coords, sizes = get_coords(m, ordered, preset=preset, emb_dir=emb_dir)
        # 데이터셋별 ages, normal_mask 만들고 concat
        ages_all, norm_all = [], []
        for d in ordered:
            n = sizes[d]
            a = get_ages(d)
            if a is None:
                a_arr = np.full(n, np.nan)
            else:
                a_arr = (a[:n] if len(a) >= n
                         else np.concatenate([a, np.full(n - len(a), np.nan)]))
            ages_all.append(a_arr)
            nm = is_normal_mask(d, n)
            if nm is None:
                # 이 데이터셋은 normal 라벨 매핑 없음 — 모두 abnormal로 처리하지 않고 결측 처리
                norm_all.append(np.full(n, np.nan))
            else:
                norm_all.append(nm.astype(float))
        ages = np.concatenate(ages_all)
        norm_arr = np.concatenate(norm_all)  # 1=normal, 0=abnormal, NaN=unknown
        bin_idx = assign_age_bin_labels(ages, bins)

        for c, (lo, hi) in enumerate(bins):
            ax = axes[r, c]
            target = bin_idx == c
            rest = ~target
            if rest.any():
                ax.scatter(coords[rest, 0], coords[rest, 1], c="#DDDDDD",
                           s=max(1, s - 1), alpha=0.2, linewidths=0, zorder=1)
            # target 안에서 normal/abnormal/unknown
            tgt_idx = np.where(target)[0]
            tgt_norm = norm_arr[tgt_idx]
            unk = np.isnan(tgt_norm)
            normal = (~unk) & (tgt_norm > 0.5)
            abnormal = (~unk) & (tgt_norm <= 0.5)
            if unk.any():
                idx = tgt_idx[unk]
                ax.scatter(coords[idx, 0], coords[idx, 1], c="#888888",
                           s=s, alpha=alpha, linewidths=0, zorder=2)
            if abnormal.any():
                idx = tgt_idx[abnormal]
                ax.scatter(coords[idx, 0], coords[idx, 1], c=neg_c,
                           s=s, alpha=alpha, linewidths=0, zorder=3)
            if normal.any():
                idx = tgt_idx[normal]
                ax.scatter(coords[idx, 0], coords[idx, 1], c=pos_c,
                           s=s, alpha=alpha, linewidths=0, zorder=4)

            # 타이틀/라벨
            n_n = int(normal.sum()); n_a = int(abnormal.sum())
            if r == 0:
                hi_l = int(hi) if hi < 200 else "+"
                ax.set_title(f"{int(lo)}–{hi_l}", fontweight="bold",
                             fontsize=11, pad=4)
            ax.text(0.98, 0.02, f"N {n_n}  A {n_a}",
                    transform=ax.transAxes, fontsize=7,
                    ha="right", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.75,
                              edgecolor="none", pad=2))
            if c == 0:
                ax.set_ylabel(m, fontweight="bold", fontsize=11,
                              rotation=0, labelpad=8, ha="right", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            # 옅은 테두리 (cell 구분)
            ax.patch.set_facecolor("#FAFAFA")

    # 단일 범례 (figure 상단 중앙)
    handles = [
        mpatches.Patch(facecolor=pos_c, label="Normal"),
        mpatches.Patch(facecolor=neg_c, label="Abnormal"),
        mpatches.Patch(facecolor="#DDDDDD", label="other ages"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=10, ncol=3,
               framealpha=0.85, frameon=False)
    title_models = (f"{len(models)} models" if len(models) > 3
                    else "+".join(models))
    fig.suptitle(
        f"{title_models} · {'+'.join(ordered)} · age × N/A · {preset}",
        fontsize=12, fontweight="bold", y=1.025,
    )
    if show:
        plt.show()
    return fig


# ════════════════════════════════════════════════════════════════════════
# High-level Notebook API
# ════════════════════════════════════════════════════════════════════════
class Explorer:
    """노트북에서 한 인스턴스로 여러 view를 빠르게 그려보기 위한 facade."""

    def __init__(self, emb_dir, log_level=logging.WARNING, ds_label_info=None,
                 exclude_datasets: Optional[Iterable[str]] = None,
                 exclude_models: Optional[Iterable[str]] = None):
        emb_dir = Path(emb_dir).resolve()
        if (emb_dir / "embeddings").is_dir():
            emb_dir = emb_dir / "embeddings"
        self.emb_dir = emb_dir
        if not emb_dir.exists():
            raise FileNotFoundError(emb_dir)
        logging.basicConfig(level=log_level)

        self.models, self.datasets, self.model_info = discover_models_and_datasets(emb_dir)
        excl_ds = set(DEFAULT_EXCLUDED_DATASETS if exclude_datasets is None
                      else exclude_datasets)
        excl_m = set(DEFAULT_EXCLUDED_MODELS if exclude_models is None
                     else exclude_models)
        self.datasets = [d for d in self.datasets if d not in excl_ds]
        self.models = [m for m in self.models if m not in excl_m]
        self.excluded_datasets = sorted(excl_ds)
        self.excluded_models = sorted(excl_m)
        # 데이터셋별 라벨 컬럼 + paper label 매트릭스 발견 여부
        self.ds_label_info = dict(ds_label_info or {})
        self.ds_to_label_cols: dict[str, list[str]] = {}
        self.ds_uses_paper_labels: dict[str, bool] = {}
        for d in self.datasets:
            # 1) paper label csv 우선
            cols = find_label_cols(d, fallback_n=0,
                                   config_info=self.ds_label_info.get(d))
            paper_csv = _resolve_label_csv(
                LABEL_CSV_HINTS.get(d) or LABEL_CSV_HINTS.get(d.lower()) or "")
            if paper_csv and cols:
                self.ds_to_label_cols[d] = cols
                self.ds_uses_paper_labels[d] = True
                continue
            # 2) paper 없으면 _labels.npy 사용 (legacy)
            n = 0
            for m in self.models:
                p = emb_dir / f"{self.model_info[m]['safe']}_{d}_labels.npy"
                if p.exists():
                    try:
                        a = np.load(p, mmap_mode="r")
                        n = a.shape[1] if a.ndim == 2 else (1 if a.ndim == 1 else 0)
                    except Exception:
                        pass
                    break
            if n:
                self.ds_to_label_cols[d] = (cols if cols else
                                             [f"label_{i}" for i in range(n)])
                self.ds_uses_paper_labels[d] = False

    # ── 정보 ────────────────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "emb_dir": str(self.emb_dir),
            "models": self.models,
            "datasets": self.datasets,
            "feature_dims": {m: self.model_info[m].get("feature_dim") for m in self.models},
            "label_cols": self.ds_to_label_cols,
            "presets": list(PRESETS.keys()),
        }

    def print_status(self):
        s = self.status()
        print(f"emb_dir: {s['emb_dir']}")
        print(f"\n모델 ({len(s['models'])}개):")
        for m in s["models"]:
            print(f"  {m}  (d={s['feature_dims'].get(m)})")
        print(f"\n데이터셋 ({len(s['datasets'])}개): {s['datasets']}")
        print(f"\n프리셋: {s['presets']}  (각 PRESETS[k] 참고)")
        for d, cols in s["label_cols"].items():
            print(f"\n[{d}] labels ({len(cols)}): {cols[:8]}{'...' if len(cols) > 8 else ''}")

    # ── 데이터 빌드 (단일 프리셋) ─────────────────────────────────────────────
    def _build(self, models, datasets, preset_name, seed=42, balance=False,
               balance_label=None, n_samples=None, color_mode="dataset"):
        preset = PRESETS[preset_name]
        ordered = sorted(datasets)
        ds_t = tuple(ordered)
        all_data: dict[str, dict] = {}
        cache_hits: dict[str, str | None] = {}

        for m in models:
            safe = self.model_info[m]["safe"]
            emb_by_ds, lbl_by_ds = {}, {}
            for d in ordered:
                e, l = _load_embeddings_cached(str(self.emb_dir), safe, d)
                if e is None:
                    continue
                # paper label이 있으면 npy 대신 사용
                if self.ds_uses_paper_labels.get(d):
                    paper_mat, _ = _load_paper_labels_aligned(d, len(e))
                    if paper_mat is not None and len(paper_mat) == len(e):
                        l = paper_mat
                if n_samples and n_samples < len(e):
                    e, l = subsample(e, l, n_samples)
                emb_by_ds[d] = e
                lbl_by_ds[d] = l
            if not emb_by_ds:
                cache_hits[m] = None
                all_data[m] = {}
                continue

            ages_by_ds = {}
            if color_mode == "age":
                for d in emb_by_ds:
                    a = _load_age_cached(d, json.dumps(self.ds_label_info.get(d, {}),
                                                      default=str))
                    if a is not None:
                        target_n = len(emb_by_ds[d])
                        if len(a) >= target_n:
                            ages_by_ds[d] = a[:target_n]
                        else:
                            ages_by_ds[d] = np.concatenate(
                                [a, np.full(target_n - len(a), np.nan)])
                    else:
                        ages_by_ds[d] = None

            # 1:1 균형
            if balance and color_mode == "label" and balance_label:
                ds_n, lbl_n = balance_label
                for d in list(emb_by_ds.keys()):
                    if d != ds_n:
                        continue
                    cols = self.ds_to_label_cols.get(d, [])
                    if lbl_n not in cols:
                        continue
                    li = cols.index(lbl_n)
                    l = lbl_by_ds.get(d)
                    if l is None or l.ndim != 2 or li >= l.shape[1]:
                        continue
                    bin_ = (l[:, li] > 0).astype(int)
                    keep = balanced_indices(bin_, seed=seed)
                    emb_by_ds[d] = emb_by_ds[d][keep]
                    lbl_by_ds[d] = l[keep]
                    if d in ages_by_ds and ages_by_ds[d] is not None:
                        ages_by_ds[d] = ages_by_ds[d][keep]

            present = [d for d in ordered if d in emb_by_ds and len(emb_by_ds[d]) > 0]
            if not present:
                cache_hits[m] = None
                all_data[m] = {}
                continue

            # 캐시 사용 여부: 균형/서브샘플 시 OFF
            try_cache = (preset["tag"] in {"orig", "cosineL2", "euclideanL2"}
                         and not balance and n_samples is None)
            save_after = try_cache
            coords_by_ds, hit = compute_combined_for_model(
                self.emb_dir, safe, present, emb_by_ds, preset, seed,
                try_cache=try_cache, save_after=save_after)
            cache_hits[m] = hit

            all_data[m] = {}
            for d in present:
                e = emb_by_ds[d].astype(np.float32)
                metrics_emb = l2_normalize(e) if preset["do_l2"] else e
                all_data[m][d] = (coords_by_ds[d], metrics_emb,
                                  lbl_by_ds.get(d), ages_by_ds.get(d) if color_mode == "age" else None)

        return all_data, cache_hits, preset

    # ── public: 단일 view ───────────────────────────────────────────────────
    def view(self, models=None, datasets=None, preset="orig",
             color_mode="dataset", labels=None, age_bins="0,18,30,40,50,60,70,80,200",
             balance_one_to_one=False, balance_label=None,
             seed=42, n_samples=None,
             point_size=4, point_alpha=0.5, fig_dpi=120, show=True):
        """
        models: list[str] or None (전체)
        datasets: list[str] or None (전체)
        preset: "orig" | "cosineL2"
        color_mode: "dataset" | "label" | "age"
        labels: [(ds_name, label_col), ...] (color_mode='label')
        age_bins: "0,18,40,60,80,200" 또는 [0,18,40,...] (color_mode='age')
        balance_one_to_one: True면 balance_label 기준 1:1 다운샘플
        balance_label: (ds_name, label_col)
        n_samples: int or None (None=전체)
        """
        models = list(models) if models else self.models
        datasets = list(datasets) if datasets else self.datasets
        if labels is None:
            labels = []
            if color_mode == "label":
                # 기본: 각 데이터셋의 첫 라벨
                for d in datasets:
                    cols = self.ds_to_label_cols.get(d, [])
                    if cols:
                        labels.append((d, cols[0]))

        bins = parse_age_bins(age_bins) if color_mode == "age" else []

        all_data, cache_hits, preset_d = self._build(
            models, datasets, preset, seed=seed,
            balance=balance_one_to_one, balance_label=balance_label,
            n_samples=n_samples, color_mode=color_mode,
        )

        # pos_neg_names
        pos_neg_names = {}
        if color_mode == "label":
            for d in datasets:
                info = self.ds_label_info.get(d, {})
                pos_n = info.get("positive_label", "Positive")
                neg_n = info.get("negative_label", "Negative")
                cols = self.ds_to_label_cols.get(d, [])
                for i, col in enumerate(cols):
                    if (d, col) in labels:
                        pos_neg_names[(d, col)] = (i, pos_n, neg_n)

        umap_params = dict(
            n_neighbors=preset_d["n_neighbors"], min_dist=preset_d["min_dist"],
            metric=preset_d["metric"], do_l2=preset_d["do_l2"], seed=seed,
        )
        n_hit = sum(1 for v in cache_hits.values() if v)
        n_total = len(cache_hits)
        cache_str = f"cache {n_hit}/{n_total}"
        title_prefix = f"[{preset_d['tag']}{' bal' if balance_one_to_one else ''} | {cache_str}] "
        fig, metrics = render_figure(
            selected_models=models, selected_datasets=sorted(datasets),
            selected_labels=labels if color_mode == "label" else [],
            all_data=all_data, umap_params=umap_params,
            color_mode=color_mode, pos_neg_names=pos_neg_names,
            age_bins=bins, fig_dpi=fig_dpi,
            point_size=point_size, point_alpha=point_alpha,
            title_prefix=title_prefix,
        )
        if show and fig is not None:
            plt.show()
        return fig, metrics

    # ── public: Plotly (WebGL) 빠른 인터랙티브 ───────────────────────────────
    def view_plotly(self, models=None, datasets=None, preset="orig",
                    color_mode="dataset", labels=None,
                    age_bins="0,18,40,60,80,200",
                    balance_one_to_one=False, balance_label=None,
                    seed=42, n_samples=None,
                    point_size=3, opacity=0.6,
                    height_per_row=240, width_per_col=320,
                    show=True):
        """
        Plotly + WebGL(Scattergl)로 grid 시각화. matplotlib보다 훨씬 빠르고
        zoom/pan/legend toggle 가능. 노트북 inline 표시.
        반환: plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise RuntimeError("plotly가 필요합니다: pip install plotly") from e

        models = list(models) if models else self.models
        datasets = list(datasets) if datasets else self.datasets
        bins = parse_age_bins(age_bins) if color_mode == "age" else []

        all_data, cache_hits, preset_d = self._build(
            models, datasets, preset, seed=seed,
            balance=balance_one_to_one, balance_label=balance_label,
            n_samples=n_samples, color_mode=color_mode,
        )
        ordered_ds = sorted(datasets)

        # column_titles / pos_neg / col 정의
        if color_mode == "dataset":
            col_keys = ordered_ds
            col_titles = ordered_ds
        elif color_mode == "label":
            if not labels:
                labels = []
                for d in ordered_ds:
                    cs = self.ds_to_label_cols.get(d, [])
                    if cs:
                        labels.append((d, cs[0]))
            col_keys = labels
            col_titles = [f"{d}/{c}" for d, c in labels]
        else:  # age
            col_keys = list(range(len(bins)))
            col_titles = [f"{int(lo)}–{int(hi) if hi < 200 else '+'}"
                          for lo, hi in bins]

        n_rows = len(models)
        n_cols = max(1, len(col_keys))
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            column_titles=col_titles,
            row_titles=models,
            horizontal_spacing=0.015, vertical_spacing=0.04,
            shared_xaxes=False, shared_yaxes=False,
        )

        for r, mname in enumerate(models, start=1):
            md = all_data.get(mname, {})

            if color_mode == "dataset":
                for c, dname in enumerate(ordered_ds, start=1):
                    e = md.get(dname)
                    if e is None:
                        continue
                    cds = e[0]
                    # 배경: 다른 데이터셋들을 한 trace로 통합 (trace 수 절감)
                    bg_parts = [md[d2][0] for d2 in ordered_ds
                                if d2 != dname and md.get(d2) is not None]
                    if bg_parts:
                        bg = np.concatenate(bg_parts, axis=0)
                        fig.add_trace(go.Scattergl(
                            x=bg[:, 0], y=bg[:, 1], mode="markers",
                            marker=dict(color="#DDDDDD",
                                        size=max(1, point_size - 1),
                                        opacity=0.2),
                            showlegend=False, hoverinfo="skip",
                        ), row=r, col=c)
                    # 강조
                    color = PALETTE[ordered_ds.index(dname) % len(PALETTE)]
                    fig.add_trace(go.Scattergl(
                        x=cds[:, 0], y=cds[:, 1], mode="markers",
                        marker=dict(color=color, size=point_size, opacity=opacity),
                        name=f"{dname} (n={len(cds):,})",
                        showlegend=(r == 1), legendgroup=dname,
                    ), row=r, col=c)

            elif color_mode == "label":
                for c, (dname, lname) in enumerate(col_keys, start=1):
                    e = md.get(dname)
                    if e is None or e[2] is None:
                        continue
                    cols = self.ds_to_label_cols.get(dname, [])
                    if lname not in cols:
                        continue
                    li = cols.index(lname)
                    if li >= e[2].shape[1]:
                        continue
                    bin_ = (e[2][:, li] > 0).astype(int)
                    pmask = bin_ == 1
                    cds = e[0]
                    fig.add_trace(go.Scattergl(
                        x=cds[~pmask, 0], y=cds[~pmask, 1], mode="markers",
                        marker=dict(color="#e74c3c", size=point_size, opacity=opacity),
                        name=f"{lname} neg",
                        showlegend=(r == 1 and c == 1), legendgroup="neg",
                    ), row=r, col=c)
                    fig.add_trace(go.Scattergl(
                        x=cds[pmask, 0], y=cds[pmask, 1], mode="markers",
                        marker=dict(color="#2ecc71", size=point_size, opacity=opacity),
                        name=f"{lname} pos",
                        showlegend=(r == 1 and c == 1), legendgroup="pos",
                    ), row=r, col=c)

            else:  # age
                # 모든 데이터셋의 좌표/나이 합치기
                all_coords, all_ages = [], []
                for d in ordered_ds:
                    e = md.get(d)
                    if e is None or len(e) < 4 or e[3] is None: continue
                    all_coords.append(e[0]); all_ages.append(e[3])
                if not all_coords:
                    continue
                coords = np.concatenate(all_coords, axis=0)
                ages = np.concatenate(all_ages, axis=0)
                bidx = assign_age_bin_labels(ages, bins)
                for c, (lo, hi) in enumerate(bins, start=1):
                    target = bidx == (c - 1)
                    rest = ~target
                    if rest.any():
                        fig.add_trace(go.Scattergl(
                            x=coords[rest, 0], y=coords[rest, 1], mode="markers",
                            marker=dict(color="#DDDDDD",
                                        size=max(1, point_size - 1), opacity=0.25),
                            showlegend=False, hoverinfo="skip",
                        ), row=r, col=c)
                    color = AGE_PALETTE[(c - 1) % len(AGE_PALETTE)]
                    fig.add_trace(go.Scattergl(
                        x=coords[target, 0], y=coords[target, 1], mode="markers",
                        marker=dict(color=color, size=point_size, opacity=opacity),
                        name=f"{int(lo)}–{int(hi) if hi < 200 else '+'} (n={int(target.sum()):,})",
                        showlegend=(r == 1), legendgroup=f"age_{c}",
                    ), row=r, col=c)

        # axes 정리: 눈금/그리드 제거
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                         scaleanchor=None)
        fig.update_layout(
            height=max(300, height_per_row * n_rows + 80),
            width=max(400, width_per_col * n_cols + 200),
            margin=dict(l=80, r=20, t=70, b=20),
            title=(f"<b>UMAP [{preset_d['tag']}"
                   f"{' bal' if balance_one_to_one else ''}]</b>  "
                   f"n_neighbors={preset_d['n_neighbors']}, "
                   f"min_dist={preset_d['min_dist']}, "
                   f"metric={preset_d['metric']}, "
                   f"L2={'on' if preset_d.get('do_l2') else 'off'}"),
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=10)),
        )
        if show:
            fig.show()
        return fig

    # ── public: Plotly + ipywidgets 인터랙티브 ───────────────────────────────
    def interact_plotly(self):
        """plotly + ipywidgets 컨트롤. 컨트롤 변경 시 즉시 plotly figure 갱신."""
        try:
            import ipywidgets as W
            from IPython.display import display, clear_output
        except ImportError as e:
            raise RuntimeError("ipywidgets가 필요합니다") from e

        m_select = W.SelectMultiple(options=self.models, value=tuple(self.models),
                                    description="models",
                                    rows=min(9, len(self.models)))
        d_select = W.SelectMultiple(options=self.datasets, value=tuple(self.datasets),
                                    description="datasets",
                                    rows=min(6, len(self.datasets)))
        preset = W.RadioButtons(options=list(PRESETS.keys()), value="orig",
                                description="preset")
        color = W.RadioButtons(options=["dataset", "label", "age"],
                               value="dataset", description="color")
        age_bins_w = W.Text(value="0,18,40,60,80,200", description="age bins")
        label_pairs = [(f"{d}/{c}", (d, c))
                       for d in self.datasets
                       for c in self.ds_to_label_cols.get(d, [])]
        labels_w = W.Dropdown(options=label_pairs, description="label")
        balance_w = W.Checkbox(value=False, description="1:1 balance")
        ps_w = W.IntSlider(value=3, min=1, max=12, description="size")
        op_w = W.FloatSlider(value=0.6, min=0.1, max=1.0, step=0.05,
                              description="opacity")
        out = W.Output()

        def render(_=None):
            with out:
                clear_output(wait=True)
                bal_lbl = labels_w.value if balance_w.value else None
                lbls = [labels_w.value] if (color.value == "label"
                                            and labels_w.value) else None
                self.view_plotly(
                    models=list(m_select.value) or None,
                    datasets=list(d_select.value) or None,
                    preset=preset.value,
                    color_mode=color.value,
                    labels=lbls,
                    age_bins=age_bins_w.value,
                    balance_one_to_one=balance_w.value,
                    balance_label=bal_lbl,
                    point_size=ps_w.value, opacity=op_w.value,
                    show=True,
                )

        for w in (m_select, d_select, preset, color, age_bins_w, labels_w,
                  balance_w, ps_w, op_w):
            w.observe(render, names="value")

        ui = W.HBox([
            W.VBox([m_select, d_select, preset]),
            W.VBox([color, labels_w, balance_w, age_bins_w]),
            W.VBox([ps_w, op_w]),
        ])
        display(ui, out)
        render()

    # ── public: 두 프리셋 나란히 ──────────────────────────────────────────────
    def compare(self, presets=("orig", "cosineL2"), models=None, datasets=None,
                color_mode="dataset", labels=None,
                age_bins="0,18,30,40,50,60,70,80,200",
                balance_one_to_one=False, balance_label=None,
                seed=42, n_samples=None,
                point_size=4, point_alpha=0.5, fig_dpi=120, show=True):
        """여러 프리셋의 figure를 차례로 표시 (각각 별도 figure)."""
        figs = []
        for p in presets:
            f, _ = self.view(
                models=models, datasets=datasets, preset=p,
                color_mode=color_mode, labels=labels, age_bins=age_bins,
                balance_one_to_one=balance_one_to_one, balance_label=balance_label,
                seed=seed, n_samples=n_samples,
                point_size=point_size, point_alpha=point_alpha, fig_dpi=fig_dpi,
                show=show,
            )
            figs.append(f)
        return figs

    # ── public: ipywidgets 인터랙티브 UI ──────────────────────────────────────
    def interact(self):
        """ipywidgets 기반 컨트롤 패널을 띄운다 (Jupyter 전용)."""
        try:
            import ipywidgets as W
            from IPython.display import display, clear_output
        except ImportError as e:
            raise RuntimeError("ipywidgets가 필요합니다: pip install ipywidgets") from e

        m_select = W.SelectMultiple(options=self.models, value=tuple(self.models),
                                    description="models", rows=min(9, len(self.models)))
        d_select = W.SelectMultiple(options=self.datasets, value=tuple(self.datasets),
                                    description="datasets", rows=min(6, len(self.datasets)))
        preset = W.RadioButtons(options=list(PRESETS.keys()), value="orig",
                                description="preset")
        color = W.RadioButtons(options=["dataset", "label", "age"],
                               value="dataset", description="color")
        age_bins_w = W.Text(value="0,18,30,40,50,60,70,80,200", description="age bins")

        # 라벨 옵션 (선택된 datasets에서 동적으로 갱신)
        label_pairs = [(f"{d}/{c}", (d, c))
                       for d in self.datasets
                       for c in self.ds_to_label_cols.get(d, [])]
        labels_w = W.SelectMultiple(options=label_pairs,
                                    value=tuple(),
                                    description="labels", rows=8)
        balance_w = W.Checkbox(value=False, description="1:1 balance")
        balance_lbl_w = W.Dropdown(options=label_pairs, description="balance label")

        n_neighbors = W.IntSlider(value=15, min=5, max=100, step=5,
                                   description="n_neighbors", disabled=True)
        seed_w = W.IntText(value=42, description="seed")
        ps_w = W.IntSlider(value=4, min=1, max=15, description="point size")
        pa_w = W.FloatSlider(value=0.5, min=0.1, max=1.0, step=0.05, description="alpha")

        out = W.Output()

        def render(_=None):
            with out:
                clear_output(wait=True)
                models = list(m_select.value) or None
                datasets = list(d_select.value) or None
                lbls = list(labels_w.value) if color.value == "label" else None
                bal_lbl = balance_lbl_w.value if balance_w.value else None
                self.view(
                    models=models, datasets=datasets, preset=preset.value,
                    color_mode=color.value, labels=lbls,
                    age_bins=age_bins_w.value,
                    balance_one_to_one=balance_w.value, balance_label=bal_lbl,
                    seed=seed_w.value,
                    point_size=ps_w.value, point_alpha=pa_w.value,
                )

        for w in (m_select, d_select, preset, color, age_bins_w, labels_w,
                  balance_w, balance_lbl_w, seed_w, ps_w, pa_w):
            w.observe(render, names="value")

        controls_left = W.VBox([m_select, d_select, preset])
        controls_mid = W.VBox([color, labels_w, balance_w, balance_lbl_w])
        controls_right = W.VBox([age_bins_w, seed_w, ps_w, pa_w])
        ui = W.HBox([controls_left, controls_mid, controls_right])

        display(ui, out)
        render()
