"""
모든 (모델 × 프리셋) UMAP 좌표 캐시 사전 생성
=================================================
기본은 PTBXL+ZZU concat → 단일 UMAP fit, 좌표를 .npy로 저장.
explorer는 이 파일이 있으면 즉시 로드하고, 없으면 그때 계산해서 저장한다.

실행:
  python scripts/precompute_umap_cache.py
  python scripts/precompute_umap_cache.py --emb_dir results/embeddings
  python scripts/precompute_umap_cache.py --presets orig cosineL2
  python scripts/precompute_umap_cache.py --skip-existing

저장 파일명:
  - 원본(orig):     {safe}_umap_coords.npy
  - cosine+L2:     {safe}_umap_coords_cosineL2.npy
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from umap import UMAP


def sanitize(name: str) -> str:
    return (name.replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace("/", "_"))


MODEL_ORDER_HINT = [
    "CPC", "ECG-FM", "ECG-FM-KED", "ECG-Founder", "ECG-JEPA",
    "HuBERT-ECG", "MERL (ResNet)", "MERL (ViT)", "ST-MEM",
]

PRESETS = {
    "orig": dict(
        n_neighbors=15, min_dist=0.1, metric="euclidean", do_l2=False,
    ),
    "euclideanL2": dict(
        n_neighbors=15, min_dist=0.1, metric="euclidean", do_l2=True,
    ),
    "cosineL2": dict(
        n_neighbors=30, min_dist=0.05, metric="cosine", do_l2=True,
    ),
}


def cache_filename_for(safe: str, preset_tag: str, ds_tuple: tuple) -> str:
    """umap_explorer.cache_filenames_for 와 동일한 규칙으로 저장 파일명 생성."""
    is_legacy = tuple(ds_tuple) == ("ptbxl", "zzu")
    suffix = "" if is_legacy else "__" + "_".join(ds_tuple)
    if preset_tag == "orig":
        return f"{safe}_umap_coords{suffix}.npy"
    if preset_tag == "cosineL2":
        return f"{safe}_umap_coords_cosineL2{suffix}.npy"
    if preset_tag == "euclideanL2":
        return f"{safe}_umap_coords_euclideanL2{suffix}.npy"
    return ""


def discover_models_and_datasets(emb_dir: Path):
    models: dict[str, str] = {}  # display -> safe
    datasets: set[str] = set()

    for f in emb_dir.glob("*_meta.npz"):
        safe = f.stem.replace("_meta", "")
        display = next((m for m in MODEL_ORDER_HINT if sanitize(m) == safe), safe)
        models[display] = safe

    known_safes = set(models.values())
    for npy in emb_dir.glob("*.npy"):
        stem = npy.stem
        if stem.endswith("_labels") or "_umap_" in stem:
            continue
        for safe in sorted(known_safes, key=len, reverse=True):
            if stem.startswith(safe + "_"):
                datasets.add(stem[len(safe) + 1:])
                break

    sorted_models = sorted(
        models.items(),
        key=lambda kv: (MODEL_ORDER_HINT.index(kv[0]) if kv[0] in MODEL_ORDER_HINT else 999, kv[0])
    )
    return sorted_models, sorted(datasets)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


def main():
    p = argparse.ArgumentParser(description="UMAP 좌표 캐시 사전 생성")
    p.add_argument("--emb_dir", type=str,
                   default=str(PROJECT_ROOT / "results" / "embeddings"))
    p.add_argument("--presets", nargs="+", default=list(PRESETS.keys()),
                   choices=list(PRESETS.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing", action="store_true",
                   help="이미 저장된 캐시는 재계산하지 않음 (default: 항상 덮어씀)")
    p.add_argument("--exclude", nargs="*", default=["mimic4"],
                   help="제외할 데이터셋 (default: mimic4)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    emb_dir = Path(args.emb_dir).resolve()
    if not emb_dir.exists():
        logging.error(f"임베딩 디렉토리 없음: {emb_dir}")
        sys.exit(1)

    models, datasets = discover_models_and_datasets(emb_dir)
    excl = set(args.exclude or [])
    datasets = [d for d in datasets if d not in excl]
    if not models or not datasets:
        logging.error("모델 또는 데이터셋을 찾지 못함")
        sys.exit(1)
    if excl:
        logging.info(f"제외된 데이터셋: {sorted(excl)}")
    ordered_ds = sorted(datasets)
    logging.info(f"모델 {len(models)}개, 데이터셋 {ordered_ds}")
    logging.info(f"프리셋: {args.presets}")

    for display, safe in models:
        # 임베딩 로드
        emb_by_ds = {}
        for d in ordered_ds:
            ep = emb_dir / f"{safe}_{d}.npy"
            if not ep.exists():
                continue
            emb_by_ds[d] = np.load(ep)
        present = [d for d in ordered_ds if d in emb_by_ds]
        if not present:
            logging.warning(f"  {display}: 임베딩 없음, skip")
            continue
        sizes = {d: len(emb_by_ds[d]) for d in present}
        logging.info(f"\n=== {display} (safe={safe}) — datasets={present}, "
                     f"sizes={sizes} ===")

        for preset_name in args.presets:
            preset = PRESETS[preset_name]
            save_fn = cache_filename_for(safe, preset_name, tuple(present))
            target = emb_dir / save_fn
            if args.skip_existing and target.exists():
                logging.info(f"  [{preset_name}] {target.name} — 이미 존재, skip")
                continue

            parts = []
            for d in present:
                e = emb_by_ds[d].astype(np.float32)
                if preset["do_l2"]:
                    e = l2_normalize(e)
                parts.append(e)
            concat = np.concatenate(parts, axis=0)
            logging.info(f"  [{preset_name}] UMAP fit (N={len(concat)}, "
                         f"d={concat.shape[1]}) ...")
            reducer = UMAP(
                n_components=2,
                n_neighbors=preset["n_neighbors"],
                min_dist=preset["min_dist"],
                metric=preset["metric"],
                random_state=args.seed,
            )
            coords = reducer.fit_transform(concat)
            np.save(target, coords)
            logging.info(f"  [{preset_name}] ✓ saved → {target.name} "
                         f"(shape={coords.shape})")

    logging.info("\n완료.")


if __name__ == "__main__":
    main()
