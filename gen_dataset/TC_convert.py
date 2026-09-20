#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import yaml
import pandas as pd
import numpy as np
import importlib.util
from typing import Optional, Dict, Any
from collections.abc import Sequence


# Conversion settings.
GLOBAL_CONFIG_PATH = "./config.yaml"
INPUT_CENTERS_PY  = "./aaa.py"
OUTPUT_PATH       = "./baseline_o.yaml"
START_COL_NAME    = None
INCLUDE_DEBUG     = True
# Numeric values within this tolerance are treated as duplicates.
DEDUP_TOL         = 1e-6


def load_global_config(path: Optional[str]) -> Dict[str, Any]:
    print(f"[DEBUG] Loading global configuration: {path}")
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Global configuration not found: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"[DEBUG] Loaded config.yaml: {cfg}")
    required = ["DATA_BASE_DIR", "FILE_EXTENSION"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")
    return cfg


def _as_dict_type_centers(tc_raw):
    """Normalize TYPE_CENTERS from a dict or a sequence containing a dict."""
    if isinstance(tc_raw, dict):
        return tc_raw
    if isinstance(tc_raw, Sequence) and not isinstance(tc_raw, (str, bytes)):
        for item in tc_raw:
            if isinstance(item, dict):
                return item
    raise TypeError(
        f"TYPE_CENTERS must be a dict or a sequence containing one; got {type(tc_raw).__name__}"
    )


def _dedup_numeric_list_keep_order(seq, tol=1e-6):
    """Deduplicate numeric-like values while preserving their order."""
    out = []
    seen = []
    for x in seq or []:
        try:
            xf = float(x)
            is_float_like = True
        except Exception:
            xf = x
            is_float_like = False

        dup = False
        for y in seen:
            if is_float_like and isinstance(y, float):
                if abs(xf - y) <= tol:
                    dup = True
                    break
            else:
                if xf == y:
                    dup = True
                    break

        if not dup:
            out.append(x)
            seen.append(xf if is_float_like else xf)
    return out


def _dedup_type_centers_inplace(tc: dict, tol=1e-6, verbose=True):
    """Deduplicate positive and negative centers in place."""
    for st, chroms in tc.items():
        if not isinstance(chroms, dict):
            continue
        for chrom, pn in chroms.items():
            if not isinstance(pn, dict):
                continue
            pos = pn.get("pos", []) or []
            neg = pn.get("neg", []) or []

            pos_dedup = _dedup_numeric_list_keep_order(pos, tol=tol)
            neg_dedup = _dedup_numeric_list_keep_order(neg, tol=tol)

            if verbose and (len(pos_dedup) != len(pos) or len(neg_dedup) != len(neg)):
                print(f"[INFO] Deduplicated {st} {chrom}: "
                      f"pos {len(pos)} -> {len(pos_dedup)}, "
                      f"neg {len(neg)} -> {len(neg_dedup)}")

            pn["pos"] = pos_dedup
            pn["neg"] = neg_dedup
def load_type_centers_from_py(py_path: str) -> Dict[str, Any]:
    print(f"[DEBUG] Loading input file: {py_path}")
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Input file not found: {py_path!r}")

    spec = importlib.util.spec_from_file_location("centers_module", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    if not hasattr(mod, "TYPE_CENTERS"):
        raise KeyError("The input file does not define TYPE_CENTERS")

    tc_raw = getattr(mod, "TYPE_CENTERS")
    print(f"[DEBUG] Raw TYPE_CENTERS type: {type(tc_raw).__name__}")

    tc = _as_dict_type_centers(tc_raw)

    _dedup_type_centers_inplace(tc, tol=DEDUP_TOL, verbose=True)

    print(f"[DEBUG] Normalized TYPE_CENTERS keys: {list(tc.keys())}")
    return {"TYPE_CENTERS": tc}


def merge_type_chrom_data(base_dir: str,
                          sample_type: str,
                          file_ext: str,
                          start_col_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    type_dir = os.path.join(base_dir, sample_type)
    file_list = glob.glob(os.path.join(type_dir, f"*{file_ext}"))
    print(f"[DEBUG] Found {len(file_list)} {sample_type} files in {type_dir}: {file_list}")

    chrom_frames = {}
    for fp in file_list:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        print(f"[DEBUG] Loaded {fp}; columns={list(df.columns)}, rows={len(df)}")

        if "chr" not in df.columns:
            print(f"[WARN] Skipping {fp}; missing column 'chr'")
            continue

        if start_col_name and start_col_name in df.columns:
            start_series = pd.to_numeric(df[start_col_name], errors="coerce")
            print(f"[DEBUG] Using {start_col_name} as the start column")
        else:
            first_col = df.columns[0]
            start_series = pd.to_numeric(df[first_col], errors="coerce")
            print(f"[DEBUG] Using first column {first_col} as start")

        df = df.copy()
        df["__start_bp__"] = start_series

        for chrom in df["chr"].unique():
            chrom_frames.setdefault(chrom, []).append(df[df["chr"] == chrom])

    merged = {}
    for chrom, parts in chrom_frames.items():
        df_chrom = pd.concat(parts, ignore_index=True)
        starts = df_chrom["__start_bp__"].astype(float).values
        valid = ~np.isnan(starts)
        if not valid.any():
            print(f"[WARN] Skipping {sample_type} {chrom}; all start values are NaN")
            continue
        df_chrom = df_chrom.loc[valid].reset_index(drop=True)
        starts = starts[valid]
        print(f"[DEBUG] Merged {sample_type} {chrom}; rows={len(df_chrom)}")
        merged[chrom] = {"df": df_chrom, "starts": starts}

    return merged


def mb_to_index(starts_bp: np.ndarray, center_mb: float) -> int:
    target_bp = float(center_mb) * 1e6
    idx = int(np.argmin(np.abs(starts_bp - target_bp)))
    return idx


def convert_centers_to_indices(global_cfg: Dict[str, Any],
                               centers_cfg: Dict[str, Any],
                               start_col_name: Optional[str] = None,
                               include_debug: bool = False) -> Dict[str, Any]:
    base_dir = global_cfg["DATA_BASE_DIR"]
    file_ext = global_cfg["FILE_EXTENSION"]
    type_centers = centers_cfg.get("TYPE_CENTERS", {})

    out: Dict[str, Any] = {"VALUE_FORMAT": "index", "TYPE_CENTERS": {}}
    if include_debug:
        out["DEBUG_META"] = {}

    print(f"[DEBUG] Processing {len(type_centers)} types: {list(type_centers.keys())}")

    for sample_type, chrom_map in type_centers.items():
        print(f"[INFO] Processing type: {sample_type}")
        merged = merge_type_chrom_data(base_dir, sample_type, file_ext, start_col_name=start_col_name)
        out["TYPE_CENTERS"].setdefault(sample_type, {})
        if include_debug:
            out["DEBUG_META"].setdefault(sample_type, {})

        for chrom, lists in chrom_map.items():
            print(f"[DEBUG] Processing {sample_type} {chrom}; pos={len(lists.get('pos', []))}, neg={len(lists.get('neg', []))}")
            if chrom not in merged:
                print(f"[WARN] No data for {sample_type} {chrom}; writing empty lists")
                out["TYPE_CENTERS"][sample_type][chrom] = {"pos": [], "neg": []}
                if include_debug:
                    out["DEBUG_META"][sample_type][chrom] = {"pos": [], "neg": []}
                continue

            starts = merged[chrom]["starts"]
            pos_mb = lists.get("pos", []) or []
            neg_mb = lists.get("neg", []) or []

            pos_idx, neg_idx = [], []
            pos_dbg, neg_dbg = [], []

            for mb in pos_mb:
                idx = mb_to_index(starts, mb)
                pos_idx.append(int(idx))
                if include_debug:
                    pos_dbg.append({"orig_mb": float(mb), "matched_bp": float(starts[idx])})

            for mb in neg_mb:
                idx = mb_to_index(starts, mb)
                neg_idx.append(int(idx))
                if include_debug:
                    neg_dbg.append({"orig_mb": float(mb), "matched_bp": float(starts[idx])})

            out["TYPE_CENTERS"][sample_type][chrom] = {"pos": pos_idx, "neg": neg_idx}
            if include_debug:
                out["DEBUG_META"][sample_type][chrom] = {"pos": pos_dbg, "neg": neg_dbg}

    return out


def report_type_centers_counts(type_centers: dict):
    """Print positive, negative, and total center counts by cancer type."""
    print("\n=== Center counts by cancer type after deduplication ===")
    for cancer_type, chroms in type_centers.items():
        pos_count = sum(len(v.get("pos", [])) for v in chroms.values())
        neg_count = sum(len(v.get("neg", [])) for v in chroms.values())
        total = pos_count + neg_count
        print(f"{cancer_type:10s} -> pos: {pos_count:4d}, neg: {neg_count:4d}, total: {total:4d}")


def main():
    global_cfg = load_global_config(GLOBAL_CONFIG_PATH)
    centers_cfg = load_type_centers_from_py(INPUT_CENTERS_PY)

    report_type_centers_counts(centers_cfg["TYPE_CENTERS"])

    result = convert_centers_to_indices(
        global_cfg,
        centers_cfg,
        start_col_name=START_COL_NAME,
        include_debug=INCLUDE_DEBUG
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
