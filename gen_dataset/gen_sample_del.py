"""Generate labeled multi-scale deletion samples from YAML configuration."""

import os
import glob
import shutil
import traceback
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise RuntimeError("Missing dependency: PyYAML. Please install with `pip install pyyaml`.") from e


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def load_config(global_path: Optional[str] = None,
                type_centers_path: Optional[str] = None) -> Dict[str, Any]:
    """Load global settings and mutation-center labels from YAML files.

    ``GENSAMPLES_CONFIG`` and ``GENSAMPLES_TYPE_CENTERS`` override the default
    local files when explicit paths are not provided.
    """
    if global_path is None:
        global_path = os.environ.get("GENSAMPLES_CONFIG")
    if global_path is None:
        global_path = _first_existing(["./config_del.yaml"])
    if global_path is None:
        raise FileNotFoundError("No global config file found. Set $GENSAMPLES_CONFIG or provide ./config_del.yaml.")

    with open(global_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    centers_value_format = str(cfg.get("VALUE_FORMAT", "index")).lower()
    if type_centers_path is None:
        type_centers_path = os.environ.get("GENSAMPLES_TYPE_CENTERS")
    if type_centers_path is None:
        type_centers_path = _first_existing(["./type_centers_index_del.yaml"])
    if type_centers_path:
        with open(type_centers_path, "r", encoding="utf-8") as f:
            centers_cfg = yaml.safe_load(f) or {}
        if "TYPE_CENTERS" in centers_cfg:
            cfg["TYPE_CENTERS"] = centers_cfg["TYPE_CENTERS"]
        if "VALUE_FORMAT" in centers_cfg:
            centers_value_format = str(centers_cfg["VALUE_FORMAT"]).lower()

    required = [
        "DATA_BASE_DIR",
        "OUTPUT_DIR",
        "FILE_EXTENSION",
        "FEATURE_COL_START",
        "FIXED_FEATURE_COLS",
        "SCALE_INPUT_SHAPES",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config key(s): {missing}")

    cfg["FEATURE_COL_END"] = int(cfg["FEATURE_COL_START"]) + int(cfg["FIXED_FEATURE_COLS"])
    cfg.setdefault("TYPE_CENTERS", {})
    cfg.setdefault("NORM_METHOD", "col")
    cfg.setdefault("OUTLIER_THRESHOLD", 10.0)
    cfg.setdefault("CENTERS_VALUE_FORMAT", centers_value_format)

    cfg.setdefault("CHR_COL", "Chromosome")
    cfg.setdefault("START_COL", "Start")
    cfg.setdefault("END_COL", "End")

    cfg.setdefault("CLEAN_OUTPUT_DIR", True)

    return cfg


CONFIG = load_config()

if CONFIG.get("CLEAN_OUTPUT_DIR", True) and os.path.exists(CONFIG["OUTPUT_DIR"]):
    shutil.rmtree(CONFIG["OUTPUT_DIR"])
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

def extract_matrix_from_feature_data(
    feature_data: np.ndarray,
    center_idx: int,
    H: int,
    W: int,
    norm_method: Optional[str] = 'col',  # None, 'col', 'matrix'
    outlier_threshold: Optional[float] = 10.0
) -> np.ndarray:
    """Extract, normalize, and width-fit a centered feature window.

    Short matrices are interpolated to 2,000 rows. Boundary windows use real
    edge rows and are rolled to center the target. Columns are sorted by mean,
    padded with a derived constant, or folded round-robin to width ``W``.
    """
    feature_data = feature_data.astype(float)

    if norm_method == 'col':
        means = np.mean(feature_data, axis=0)
        stds = np.std(feature_data, axis=0)
        stds[stds == 0] = 1.0
        feature_data = (feature_data - means) / stds
    elif norm_method == 'matrix':
        mean_all = np.mean(feature_data)
        std_all = np.std(feature_data)
        std_all = std_all if std_all != 0 else 1.0
        feature_data = (feature_data - mean_all) / std_all
    elif norm_method is None:
        pass
    else:
        raise ValueError("norm_method must be one of {None, 'col', 'matrix'}.")

    if outlier_threshold is not None:
        feature_data = np.clip(feature_data, -outlier_threshold, outlier_threshold)

    num_rows, num_cols = feature_data.shape
    half = H // 2
    if num_rows < H:
        target_rows = 2000
        new_idx = np.linspace(0, num_rows - 1, num=target_rows)
        compressed = np.zeros((target_rows, num_cols), dtype=float)
        for j in range(num_cols):
            compressed[:, j] = np.interp(new_idx, np.arange(num_rows), feature_data[:, j])
        feature_data = compressed
        num_rows = target_rows

    if 0 <= center_idx - half and center_idx + half < num_rows:
        start = center_idx - half
        mat = feature_data[start:start + H]
    else:
        if center_idx < half:
            mat = feature_data[0:H]
            orig_pos = center_idx
        else:
            mat = feature_data[num_rows - H:num_rows]
            orig_pos = center_idx - (num_rows - H)
        shift = half - orig_pos
        mat = np.roll(mat, shift=shift, axis=0)

    col_means = np.mean(mat, axis=0)
    sorted_idx = np.argsort(-col_means)
    mat = mat[:, sorted_idx]

    cur_W = mat.shape[1]
    if cur_W < W:
        col_means_no_max = []
        for j in range(cur_W):
            col = mat[:, j]
            if len(col) > 1:
                col_means_no_max.append((col.sum() - col.max()) / (len(col) - 1))
            else:
                col_means_no_max.append(col.mean())
        final_mean = float(np.mean(col_means_no_max)) if len(col_means_no_max) > 0 else 0.0
        missing = W - cur_W
        pad = np.full((H, missing), final_mean, dtype=mat.dtype)
        mat = np.concatenate([mat, pad], axis=1)
    elif cur_W > W:
        combined = np.zeros((H, W), dtype=mat.dtype)
        counts = np.zeros(W, dtype=int)
        for i in range(cur_W):
            target = i % W
            combined[:, target] += mat[:, i]
            counts[target] += 1
        for j in range(W):
            if counts[j] > 0:
                combined[:, j] /= counts[j]
        mat = combined

    return mat


def write_sample_tsv(filepath: str, scale_matrices: List[np.ndarray], label: int) -> None:
    """Write the labeled multi-scale TSV sample format."""
    with open(filepath, "w") as f:
        f.write(f"label: {label}\n")
        for i, mat in enumerate(scale_matrices):
            H, W = mat.shape
            f.write(f"scale: {i}, shape: {H}x{W}\n")
            for row in mat:
                f.write("\t".join(map(str, row)) + "\n")
            f.write("\n")
    print(f"Wrote: {filepath}")


def _load_type_frames(type_dir: str,
                      file_ext: str,
                      chr_col: str,
                      start_col: str) -> Dict[str, pd.DataFrame]:
    """Load a cancer project's files and merge frames by chromosome."""
    file_list = glob.glob(os.path.join(type_dir, "*" + file_ext))
    chrom_data: Dict[str, List[pd.DataFrame]] = {}

    for filepath in file_list:
        try:
            df = pd.read_csv(filepath, sep="\t")
        except Exception as e:
            print(f"Failed to read {filepath}: {e}")
            continue

        if chr_col not in df.columns:
            print(f"Warning: skipping {filepath}; missing column '{chr_col}'.")
            continue
        if start_col not in df.columns:
            print(f"Warning: skipping {filepath}; missing column '{start_col}'.")
            continue

        for chrom in df[chr_col].dropna().unique():
            chrom_data.setdefault(str(chrom), []).append(df[df[chr_col] == chrom])

    merged: Dict[str, pd.DataFrame] = {}
    for chrom, parts in chrom_data.items():
        merged[chrom] = pd.concat(parts, ignore_index=True)

        merged[chrom][start_col] = pd.to_numeric(merged[chrom][start_col], errors="coerce")

    return merged


def _resolve_center_indices(
    centers_cfg: Dict[str, Any],
    starts: np.ndarray,
    value_format: str
) -> List[Tuple[int, float, int]]:
    """Resolve Mb coordinates or explicit indices to matrix row indices."""
    records: List[Tuple[int, float, int]] = []
    nrows = len(starts)

    if value_format == "mb":
        for val in centers_cfg or []:
            try:
                target_bp = float(val) * 1e6
            except Exception:
                continue
            idx = int(np.argmin(np.abs(starts - target_bp)))
            center_bp = int(starts[idx]) if np.isfinite(starts[idx]) else -1
            center_m = (center_bp / 1e6) if center_bp >= 0 else float("nan")
            records.append((idx, center_m, center_bp))

    elif value_format == "index":
        for val in centers_cfg or []:
            try:
                idx = int(val)
            except Exception:
                continue
            if 0 <= idx < nrows:
                center_bp = int(starts[idx]) if np.isfinite(starts[idx]) else -1
                center_m = (center_bp / 1e6) if center_bp >= 0 else float("nan")
                records.append((idx, center_m, center_bp))
            else:
                print(f"Skipping index {idx}: outside [0, {nrows-1}]")
    else:
        raise ValueError("VALUE_FORMAT must be 'mb' or 'index'.")

    return records


def generate_samples(is_output: bool = False):
    base_dir = CONFIG["DATA_BASE_DIR"]
    output_dir = CONFIG["OUTPUT_DIR"]
    file_ext = CONFIG["FILE_EXTENSION"]
    feat_start, feat_end = CONFIG["FEATURE_COL_START"], CONFIG["FEATURE_COL_END"]
    scale_shapes: List[Tuple[int, int]] = CONFIG["SCALE_INPUT_SHAPES"]

    chr_col = CONFIG["CHR_COL"]
    start_col = CONFIG["START_COL"]

    norm_method = CONFIG.get("NORM_METHOD", "col")
    outlier_thr = CONFIG.get("OUTLIER_THRESHOLD", 10.0)
    value_format = str(CONFIG.get("CENTERS_VALUE_FORMAT", "mb")).lower()

    type_counters: Dict[str, Dict[str, int]] = {}
    total_pos = 0
    total_neg = 0

    # Cache chromosome coordinates and feature matrices by cancer type.
    data_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    for type_name, chrom_dict in CONFIG["TYPE_CENTERS"].items():
        print(f"Processing type: {type_name}")
        type_counters.setdefault(type_name, {"pos": 0, "neg": 0})

        type_dir = os.path.join(base_dir, type_name)
        chrom_frames = _load_type_frames(type_dir, file_ext, chr_col, start_col)

        for chrom, df_chrom in chrom_frames.items():
            try:
                starts = df_chrom[start_col].values.astype(float)
                feature_data = df_chrom.iloc[:, feat_start:feat_end].astype(float).values
            except Exception as e:
                print(f"Failed to extract features for {type_name} {chrom}: {e}")
                continue
            data_cache[(type_name, chrom)] = (starts, feature_data)

        for chrom, centers in chrom_dict.items():
            if (type_name, chrom) not in data_cache:
                print(f"Skipping {type_name}: no data for {chrom}")
                continue

            starts, feature_data = data_cache[(type_name, chrom)]

            pos_records = _resolve_center_indices(centers.get("pos"), starts, value_format)
            neg_records = _resolve_center_indices(centers.get("neg"), starts, value_format)

            type_counters[type_name]["pos"] += len(pos_records)
            type_counters[type_name]["neg"] += len(neg_records)
            total_pos += len(pos_records)
            total_neg += len(neg_records)

            for label, records in ((1, pos_records), (0, neg_records)):
                for idx, center_m, center_bp in records:
                    scale_mats: List[np.ndarray] = []
                    ok = True
                    for (H, W) in scale_shapes:
                        try:
                            mat = extract_matrix_from_feature_data(
                                feature_data, idx, H, W,
                                norm_method=norm_method,
                                outlier_threshold=outlier_thr
                            )
                        except Exception as e:
                            class_name = "positive" if label == 1 else "negative"
                            print(f"Skipping {class_name} {type_name} {chrom} center={center_m:.6f} Mb: {e}")
                            traceback.print_exc()
                            ok = False
                            break
                        if mat.shape != (H, W):
                            print(f"Shape mismatch: got {mat.shape}, expected ({H}, {W})")
                            ok = False
                            break
                        scale_mats.append(mat)
                    if not ok:
                        continue

                    if is_output:
                        fname = f"{type_name}_{chrom}_{int(center_bp)}_{'pos' if label == 1 else 'neg'}.tsv"
                        write_sample_tsv(os.path.join(output_dir, fname), scale_mats, label)

        tp = type_counters[type_name]["pos"]
        tn = type_counters[type_name]["neg"]
        print(f"[SUMMARY] {type_name}: pos = {tp}, neg = {tn}, total = {tp + tn}")

    print(f"[TOTAL] pos = {total_pos}, neg = {total_neg}, total = {total_pos + total_neg}")

    return {
        **type_counters,
        "__TOTAL__": {"pos": total_pos, "neg": total_neg, "total": total_pos + total_neg}
    }


if __name__ == "__main__":
    generate_samples(is_output=True)
