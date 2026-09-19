# FINAL (integrated with external YAML config, INDEX-based centers)
# last modified 0429-index
from typing import Optional

import os
import glob
import numpy as np
import pandas as pd
import shutil

# =============================================================================
# Configuration loader (YAML)
# =============================================================================
try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise RuntimeError("Missing dependency: PyYAML. Please install with `pip install pyyaml`.") from e


def _first_existing(paths):
    """Return the first existing file path from the given sequence, or None."""
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def load_config(global_path: Optional[str] = None,
                type_centers_path: Optional[str] = None) -> dict:
    """
    Load and normalize configuration.
    - global_path: main config.yaml (global settings)
    - type_centers_path: optional type_centers.yaml (cancer-specific centers)
    - If not provided, will resolve via env vars or defaults.
    """
    # --- 1) Resolve global config path ---
    if global_path is None:
        global_path = os.environ.get("GENSAMPLES_CONFIG")
    if global_path is None:
        global_path = _first_existing(("./config.yaml",))
    if global_path is None:
        raise FileNotFoundError(
            "No global config file found. Set $GENSAMPLES_CONFIG or create ./config.yaml"
        )

    # --- 2) Load global config ---
    with open(global_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # --- 3) Try to load type_centers config ---
    centers_value_format = "index"  # default legacy
    if type_centers_path is None:
        type_centers_path = os.environ.get("GENSAMPLES_TYPE_CENTERS")
    if type_centers_path is None:
        # keep the previous default filename for your project
        # type_centers_path = _first_existing(("./center_position.yaml", "./type_centers.yaml"))
        type_centers_path = "./type_centers_index.yaml"
        # type_centers_path = "./rubic_sim_a.yaml"


    if type_centers_path and os.path.exists(type_centers_path):
        with open(type_centers_path, "r", encoding="utf-8") as f:
            centers_cfg = yaml.safe_load(f) or {}
        if "TYPE_CENTERS" in centers_cfg:
            cfg["TYPE_CENTERS"] = centers_cfg["TYPE_CENTERS"]
        # NEW: carry value format for centers (expect "index" in your new file)
        centers_value_format = str(centers_cfg.get("VALUE_FORMAT", centers_value_format)).lower()

    cfg["CENTERS_VALUE_FORMAT"] = centers_value_format

    # --- 4) Basic required keys ---
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

    # --- 5) Derived / aliases / defaults ---
    if "FEATURE_COL_END" not in cfg:
        cfg["FEATURE_COL_END"] = int(cfg["FEATURE_COL_START"]) + int(cfg["FIXED_FEATURE_COLS"])

    # Backward compatibility: alias row-normalization flag to column-normalization key
    if "APPLY_COLUMN_NORMALIZATION" not in cfg:
        cfg["APPLY_COLUMN_NORMALIZATION"] = bool(cfg.get("APPLY_ROW_NORMALIZATION", False))

    # Optional defaults referenced in code
    cfg.setdefault("HAS_GLOBAL", False)
    cfg.setdefault("FILL_METHOD", "mean")
    cfg.setdefault("SKIP_CONFLICTING_NEG", False)  # ignored in index mode
    cfg.setdefault("CONFLICT_DISTANCE_M", 1.0)     # ignored in index mode
    cfg.setdefault("TYPE_CENTERS", {})             # safe default

    return cfg


# Load CONFIG dict from YAML now.
CONFIG = load_config()  # Or: load_config("/absolute/path/to/config.yaml")

# =============================================================================
# Prepare output directory
# =============================================================================
# Clear output directory to ensure a clean run (same behavior as your original script)
if os.path.exists(CONFIG["OUTPUT_DIR"]):
    shutil.rmtree(CONFIG["OUTPUT_DIR"])
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# =============================================================================
# Imports for processing
# =============================================================================
from scipy.ndimage import gaussian_filter  # Keep for potential future filters (e.g., LCN, HPF)

# =============================================================================
# Core utilities
# =============================================================================

import numpy as np

import numpy as np

def extract_matrix_from_feature_data(
    feature_data, center_idx, H, W,
    norm_method='col',       # None, 'col', or 'matrix'
    outlier_threshold=10.0,  # float, e.g. 3.0 for clipping z-scores; None to disable
    norm_scope='window'        # 'full' (default) or 'window'
):
    """
    Extract an HxW window centered at row `center_idx` from `feature_data` (2D ndarray).

    New:
      - norm_scope:
          - 'full'   : normalize/clip the entire input FIRST (original behavior),
                       then interpolate (if needed), row-slice with boundary rolling,
                       column sort, and column fit.
          - 'window' : FIRST do interpolate (if needed) + row-slice (with boundary rolling),
                       THEN normalize/clip ONLY within that selected window,
                       then column sort & column fit.

    Other behaviors are identical to your original pipeline:
      - If source rows < H: resample (interpolate) to 2000 rows first (for 'full' scope,
        this happens after global normalization/clipping; for 'window' scope, before
        per-window normalization/clipping).
      - Row window extraction with boundary rolling to keep center aligned.
      - Sort columns by descending mean (within the working window).
      - If cols < W: pad with a constant column (mean_without_max aggregated across columns).
      - If cols > W: merge extra columns into W buckets by round-robin, then average.
    """
    def _normalize_inplace(A, method):
        if method == 'col':
            means = np.mean(A, axis=0)
            stds  = np.std(A, axis=0)
            stds[stds == 0] = 1.0
            A -= means
            A /= stds
        elif method == 'matrix':
            mean_all = float(np.mean(A))
            std_all  = float(np.std(A))
            if std_all == 0:
                std_all = 1.0
            A -= mean_all
            A /= std_all
        elif method is None:
            pass
        else:
            raise ValueError("norm_method must be one of {'col', 'matrix', None}.")

    def _clip_inplace(A, thr):
        if thr is not None:
            np.clip(A, -thr, thr, out=A)

    # Ensure float dtype
    feature_data = np.array(feature_data, dtype=float, copy=True)

    if norm_scope not in ('full', 'window'):
        raise ValueError("norm_scope must be 'full' or 'window'.")

    # ===== A) FULL-SCOPE NORMALIZATION/CLIPPING PATH =====
    if norm_scope == 'full':
        # 1) Pre-normalization on the WHOLE input
        _normalize_inplace(feature_data, norm_method)

        # 2) Outlier clipping (whole input)
        _clip_inplace(feature_data, outlier_threshold)

        # 3) Prepare shape + center
        num_rows, num_cols = feature_data.shape
        half = H // 2

        # If not enough rows, interpolate to 2000 rows first
        if num_rows < H:
            target_rows = 2000
            new_idx = np.linspace(0, num_rows - 1, num=target_rows)
            compressed = np.zeros((target_rows, num_cols), dtype=float)
            for j in range(num_cols):
                compressed[:, j] = np.interp(new_idx, np.arange(num_rows), feature_data[:, j])
            feature_data = compressed
            num_rows = target_rows

        # 4) Row extraction with boundary rolling to keep the center aligned
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

    # ===== B) WINDOW-SCOPE NORMALIZATION/CLIPPING PATH =====
    else:  # norm_scope == 'window'
        # 1) DO NOT normalize/clip the full matrix; first just handle rows
        num_rows, num_cols = feature_data.shape
        half = H // 2

        # If not enough rows, interpolate to 2000 rows first
        if num_rows < H:
            target_rows = 2000
            new_idx = np.linspace(0, num_rows - 1, num=target_rows)
            compressed = np.zeros((target_rows, num_cols), dtype=float)
            for j in range(num_cols):
                compressed[:, j] = np.interp(new_idx, np.arange(num_rows), feature_data[:, j])
            feature_data = compressed
            num_rows = target_rows

        # 2) Row extraction with boundary rolling to keep the center aligned
        if 0 <= center_idx - half and center_idx + half < num_rows:
            start = center_idx - half
            mat = feature_data[start:start + H].copy()
        else:
            if center_idx < half:
                mat = feature_data[0:H].copy()
                orig_pos = center_idx
            else:
                mat = feature_data[num_rows - H:num_rows].copy()
                orig_pos = center_idx - (num_rows - H)
            shift = half - orig_pos
            mat = np.roll(mat, shift=shift, axis=0)

        # 3) Now normalize/clip ONLY within the window
        _normalize_inplace(mat, norm_method)
        _clip_inplace(mat, outlier_threshold)

    # ===== Common column-side logic (unchanged) =====

    # 5) Column sort by descending mean (within the selected/processed window)
    col_means = np.mean(mat, axis=0)
    sorted_idx = np.argsort(-col_means)
    mat = mat[:, sorted_idx]

    # 6) Fit column count to W (pad or merge)
    cur_W = mat.shape[1]
    if cur_W < W:
        # Pad with a constant column derived from "mean without the max" per column
        col_means_no_max = []
        for j in range(cur_W):
            col = mat[:, j]
            col_means_no_max.append((col.sum() - col.max()) / (len(col) - 1))
        final_mean = float(np.mean(col_means_no_max)) if len(col_means_no_max) > 0 else 0.0
        missing = W - cur_W
        pad = np.full((H, missing), dtype=mat.dtype, fill_value=final_mean)
        mat = np.concatenate([mat, pad], axis=1)

    elif cur_W > W:
        # Round-robin merge extra columns into W buckets, then average
        combined = np.zeros((H, W), dtype=mat.dtype)
        counts   = np.zeros(W,   dtype=int)
        for i in range(cur_W):
            target = i % W
            combined[:, target] += mat[:, i]
            counts[target]    += 1
        for j in range(W):
            combined[:, j] /= counts[j]
        mat = combined

    # Final shape is (H, W)
    return mat




def write_sample_tsv(filepath, scale_matrices, label):
    """Write multiple matrices into a TSV-like file with a header block."""
    with open(filepath, "w") as f:
        f.write(f"label: {label}\n")
        for i, mat in enumerate(scale_matrices):
            H, W = mat.shape
            f.write(f"scale: {i}, shape: {H}x{W}\n")
            for row in mat:
                f.write("\t".join(map(str, row)) + "\n")
            f.write("\n")
    print(f"写入: {filepath}")


def compress_matrix(mat, target_rows=8000):
    """Interpolate vertically to `target_rows`, preserving number of columns."""
    original_rows = mat.shape[0]
    new_indices = np.linspace(0, original_rows - 1, num=target_rows)
    new_mat = np.zeros((target_rows, mat.shape[1]))
    for col in range(mat.shape[1]):
        new_mat[:, col] = np.interp(new_indices, np.arange(original_rows), mat[:, col])
    return new_mat


def generate_samples(is_output: bool = True):
    # ---- Read parameters from CONFIG ----
    base_dir = CONFIG["DATA_BASE_DIR"]
    file_ext = CONFIG["FILE_EXTENSION"]
    feat_start, feat_end = CONFIG["FEATURE_COL_START"], CONFIG["FEATURE_COL_END"]
    scale_shapes = CONFIG["SCALE_INPUT_SHAPES"]
    output_dir = CONFIG["OUTPUT_DIR"]

    # In index mode, conflict/m-distance is not applicable; keep flags but ignore.
    # conflict_thresh_m = float(CONFIG.get("CONFLICT_DISTANCE_M", 1.0))
    # skip_conflicting_neg = bool(CONFIG.get("SKIP_CONFLICTING_NEG", False))
    apply_col_norm = bool(CONFIG.get("APPLY_COLUMN_NORMALIZATION", False))

    # Ensure centers are index-based
    centers_format = str(CONFIG.get("CENTERS_VALUE_FORMAT", "mb")).lower()
    if centers_format != "index":
        raise ValueError(
            f"Expected VALUE_FORMAT='index' for TYPE_CENTERS, got '{centers_format}'. "
            f"Please provide the converted centers YAML (index-based)."
        )

    os.makedirs(output_dir, exist_ok=True)

    # ---------- Shared cache for merged per-(type, chrom) data ----------
    # data_cache[(type_name, chrom)] = {
    #     "df": df_chrom,
    #     "starts": <np.ndarray>,          # numeric bp coordinates for first column
    #     "feature_data": <np.ndarray>     # feature matrix slice
    # }
    data_cache = {}

    # ---------- PASS1: assemble data and resolve indices ----------
    # We will still persist a loci list for traceability; center_M/bp derived from index.
    selected_records = []  # tuple: (type_name, chrom, label, center_M, center_bp, row_index)

    # ---- NEW: counters (effective, after bounds-check) ----
    type_counters = {}   # {type_name: {"pos": int, "neg": int}}
    total_pos = 0
    total_neg = 0

    for type_name, chrom_dict in CONFIG["TYPE_CENTERS"].items():
        print(f"处理 type: {type_name}")
        type_dir = os.path.join(base_dir, type_name)
        file_list = glob.glob(os.path.join(type_dir, "*" + file_ext))
        if not file_list:
            print(f"警告: {type_dir} 无数据文件。")
            # 仍继续解析配置，但由于无数据，该 type 的计数可能为 0
        # 初始化当前 type 计数容器
        type_counters.setdefault(type_name, {"pos": 0, "neg": 0})

        # Merge all files under this type; group by 'chr'
        chrom_data = {}
        for filepath in file_list:
            try:
                df = pd.read_csv(filepath, sep="\t")
            except Exception as e:
                print(f"读取文件 {filepath} 时出错：", e)
                continue
            if "chr" not in df.columns:
                print(f"警告: 文件 {filepath} 缺少 'chr' 列，已跳过。")
                continue
            for chrom in df["chr"].unique():
                chrom_data.setdefault(chrom, []).append(df)

        for chrom in list(chrom_data.keys()):
            chrom_data[chrom] = pd.concat(chrom_data[chrom], ignore_index=True)

            # Prepare starts / feature_data
            df_chrom = chrom_data[chrom]
            try:
                # Assume the first column stores the start coordinate (bp)
                df_chrom["start"] = pd.to_numeric(df_chrom.iloc[:, 0], errors="coerce")
            except Exception as e:
                print(f"转换 {chrom} 的起始位置时出错：", e)
                del chrom_data[chrom]
                continue

            # NOTE: we take numeric values directly from the first column
            starts = df_chrom.iloc[:, 0].values.astype(float)

            try:
                feature_data = df_chrom.iloc[:, feat_start:feat_end].astype(float).values
            except Exception as e:
                print(f"提取 {chrom} 的特征矩阵区间[{feat_start}:{feat_end})失败：{e}")
                del chrom_data[chrom]
                continue

            data_cache[(type_name, chrom)] = {
                "df": df_chrom,
                "starts": starts,
                "feature_data": feature_data,
            }

        # Resolve configured indices for this type
        for chrom, centers in chrom_dict.items():
            if (type_name, chrom) not in data_cache:
                print(f"跳过: {type_name} 无 {chrom} 数据")
                continue

            starts = data_cache[(type_name, chrom)]["starts"]
            nrows = len(starts)

            # pos/neg are indices now
            pos_indices = [int(x) for x in (centers.get("pos") or [])]
            neg_indices = [int(x) for x in (centers.get("neg") or [])]

            # bounds check; derive center_bp/M from starts[idx] for traceable outputs
            for label, idx_list in ((1, pos_indices), (0, neg_indices)):
                for idx in idx_list:
                    if idx < 0 or idx >= nrows:
                        print(f"跳过: {type_name} {chrom} 索引 {idx} 超出范围 [0, {nrows-1}]")
                        continue
                    center_bp = int(starts[idx]) if np.isfinite(starts[idx]) else int(-1)
                    center_m = float(center_bp) / 1e6 if center_bp >= 0 else float("nan")
                    selected_records.append((type_name, chrom, int(label), center_m, int(center_bp), int(idx)))
                    # ---- NEW: count effective (valid) indices ----
                    if label == 1:
                        type_counters[type_name]["pos"] += 1
                        total_pos += 1
                    else:
                        type_counters[type_name]["neg"] += 1
                        total_neg += 1

        # per-type summary after resolving all chroms of this type
        tp = type_counters[type_name]["pos"]
        tn = type_counters[type_name]["neg"]
        print(f"[SUMMARY] {type_name}: pos = {tp}, neg = {tn}, total = {tp + tn}")

    # Global total after PASS1
    print(f"[TOTAL] pos = {total_pos}, neg = {total_neg}, total = {total_pos + total_neg}")

    # Persist PASS1 loci list (to-be-processed)
    # loci_tsv_path = os.path.join(output_dir, "processed_loci.tsv")
    # if selected_records:
    #     idx_df = pd.DataFrame(
    #         selected_records,
    #         columns=["cancer_type", "chrom", "label", "center_M", "center_bp", "row_index"]
    #     )
    #     idx_df.to_csv(loci_tsv_path, sep="\t", index=False)
    #     print(f"已输出位点清单（将要处理）: {loci_tsv_path}（共 {len(selected_records)} 条）")
    # else:
    #     print("无可处理位点，已结束。")
    #     return type_counters  # ---- NEW: return counters even if empty ----

    # ---------- PASS2: multi-scale extraction and write files ----------
    for (type_name, chrom, label, center_m, center_bp, idx) in selected_records:
        cache_key = (type_name, chrom)
        if cache_key not in data_cache:
            print(f"跳过: {type_name} {chrom} 数据缓存缺失（可能在 PASS1 中失败）。")
            continue

        feature_data = data_cache[cache_key]["feature_data"]

        scale_matrices = []
        skip = False

        # Multi-scale extraction
        for (H, W) in scale_shapes:
            try:
                mat = extract_matrix_from_feature_data(feature_data, idx, H, W, norm_scope='full')
            except Exception as e:
                # keep logging center_m (derived) for continuity
                print(f"跳过 {type_name} {chrom} {'正' if label==1 else '负'}例 idx={idx}：提取出错 -> {e}")
                skip = True
                break
            if mat.shape != (H, W):
                print(f"跳过 {type_name} {chrom} {'正' if label==1 else '负'}例 idx={idx}：尺寸 {mat.shape} 不符 ({H},{W})")
                skip = True
                break
            scale_matrices.append(mat)

        if skip:
            continue

        # Optional: add a horizontally compressed whole-file matrix (if flag is enabled)
        if apply_col_norm:
            try:
                full_mat = feature_data
                N = full_mat.shape[0]
                shift = (N // 2) - int(idx)
                shifted = np.roll(full_mat, shift=shift, axis=0)
                compressed = compress_matrix(shifted, target_rows=8000)
                if np.isnan(compressed).any():
                    print(f"跳过 {type_name} {chrom} {'正' if label==1 else '负'}例 idx={idx}：压缩矩阵包含 NaN")
                    continue
                scale_matrices.append(compressed)
            except Exception as e:
                print(f"跳过 {type_name} {chrom} {'正' if label==1 else '负'}例 idx={idx}：追加压缩出错 -> {e}")
                continue

        # Use center_bp (derived from index) in filename to keep the same naming scheme
        if is_output:
            filename = f"{type_name}_{chrom}_{int(center_bp)}_{'pos' if label==1 else 'neg'}.tsv"
            write_sample_tsv(os.path.join(output_dir, filename), scale_matrices, int(label))

    # ---- NEW: also return the counters for programmatic use ----
    return {
        **type_counters,
        "__TOTAL__": {"pos": total_pos, "neg": total_neg, "total": total_pos + total_neg}
    }


if __name__ == "__main__":
    generate_samples()
