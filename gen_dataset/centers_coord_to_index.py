#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert TYPE_CENTERS from coordinate values (in megabases) to row indices.

- Reads:
    1) Global config (YAML): provides DATA_BASE_DIR, FILE_EXTENSION, etc.
    2) Type centers (YAML): TYPE_CENTERS with pos/neg values in megabases.

- For each (type, chrom):
    * Merge all files under DATA_BASE_DIR/<type>/*<FILE_EXTENSION>
    * Expect a 'chr' column to separate chromosomes
    * Use the FIRST column by default as the genomic start coordinate (bp);
      or use --start-col-name to specify a named column.
    * Map each center (in Mb) to nearest row index (0-based) by start coordinate.

- Writes:
    A new YAML with:
        VALUE_FORMAT: "index"
        TYPE_CENTERS:  # same hierarchy
          <TYPE>:
            <chrN>:
              pos: [int indices...]
              neg: [int indices...]

Optionally include debug metadata (matched bp, original Mb) via --include-debug.
"""

import os
import glob
import argparse
import yaml
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_global_config(path: Optional[str]) -> Dict[str, Any]:
    """Load global config YAML and validate minimal keys."""
    if path is None:
        path = os.environ.get("GENSAMPLES_CONFIG")
    if path is None:
        path = _first_existing(("./config.yaml",))
    if path is None:
        raise FileNotFoundError(
            "No global config file found. Set $GENSAMPLES_CONFIG or create ./config.yaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required = ["DATA_BASE_DIR", "FILE_EXTENSION"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config key(s): {missing}")

    return cfg


def load_type_centers(path: Optional[str]) -> Dict[str, Any]:
    """Load centers YAML and return dict with TYPE_CENTERS."""
    if path is None:
        path = os.environ.get("GENSAMPLES_TYPE_CENTERS")
    if path is None:
        # default file name commonly used in your examples
        path = _first_existing(("./type_centers.yaml", "./center_position.yaml"))
    if path is None:
        raise FileNotFoundError(
            "No type centers file found. Set $GENSAMPLES_TYPE_CENTERS or create ./type_centers.yaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if "TYPE_CENTERS" not in d:
        raise KeyError("TYPE_CENTERS key not found in the centers YAML.")
    return d


def merge_type_chrom_data(base_dir: str,
                          cancer_type: str,
                          file_ext: str,
                          start_col_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Read and merge all files under base_dir/<cancer_type>/*<file_ext>.
    Return per-chrom dictionary with DataFrame and a numeric 'starts' np.ndarray.
    Assumes a 'chr' column exists to split chromosomes.

    If start_col_name is given, use that column as start (bp).
    Otherwise, use the FIRST column in the file as start coordinate.
    """
    type_dir = os.path.join(base_dir, cancer_type)
    file_list = glob.glob(os.path.join(type_dir, f"*{file_ext}"))

    chrom_data_frames = {}
    for fp in file_list:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception as e:
            print(f"[WARN] Failed reading file: {fp} -> {e}")
            continue

        if "chr" not in df.columns:
            print(f"[WARN] Missing 'chr' column in {fp}; skipped.")
            continue

        # Determine the start column
        if start_col_name and start_col_name in df.columns:
            start_series = pd.to_numeric(df[start_col_name], errors="coerce")
            start_idx = df.columns.get_loc(start_col_name)
        else:
            # use the first column
            first_col = df.columns[0]
            start_series = pd.to_numeric(df[first_col], errors="coerce")
            start_idx = 0

        df = df.copy()
        df["__start_bp__"] = start_series

        # Group by chromosome and store
        for chrom in df["chr"].unique():
            sub = df[df["chr"] == chrom]
            if chrom not in chrom_data_frames:
                chrom_data_frames[chrom] = [sub]
            else:
                chrom_data_frames[chrom].append(sub)

    merged = {}
    for chrom, parts in chrom_data_frames.items():
        df_chrom = pd.concat(parts, ignore_index=True)
        starts = df_chrom["__start_bp__"].astype(float).values
        # drop NaN starts (rare), while keeping indices aligned by making a filtered view
        valid_mask = ~np.isnan(starts)
        if not valid_mask.any():
            print(f"[WARN] All starts are NaN for {cancer_type} {chrom}; skipping.")
            continue
        df_chrom = df_chrom.loc[valid_mask].reset_index(drop=True)
        starts = starts[valid_mask]
        merged[chrom] = {"df": df_chrom, "starts": starts}

    return merged


def mb_to_index(starts_bp: np.ndarray, center_mb: float) -> int:
    """Find nearest index to the given center in Mb (convert to bp)."""
    target_bp = float(center_mb) * 1e6
    idx = int(np.argmin(np.abs(starts_bp - target_bp)))
    return idx


def convert_centers_to_indices(global_cfg: Dict[str, Any],
                               centers_cfg: Dict[str, Any],
                               start_col_name: Optional[str] = None,
                               include_debug: bool = False) -> Dict[str, Any]:
    """
    Convert TYPE_CENTERS pos/neg (in Mb) to indices based on nearest 'start' bp.

    Returns a dict ready to be dumped as YAML:
    {
      "VALUE_FORMAT": "index",
      "TYPE_CENTERS": { <type>: { <chr>: {pos: [...], neg: [...] } } }
      [optional "DEBUG_META": ...]
    }
    """
    base_dir = global_cfg["DATA_BASE_DIR"]
    file_ext = global_cfg["FILE_EXTENSION"]
    type_centers = centers_cfg.get("TYPE_CENTERS", {})

    out: Dict[str, Any] = {"VALUE_FORMAT": "index", "TYPE_CENTERS": {}}
    if include_debug:
        out["DEBUG_META"] = {}  # store matched bp and original mb for auditing

    for cancer_type, chrom_map in type_centers.items():
        print(f"[INFO] processing type={cancer_type}")
        # build per-chrom merged data for this type once
        merged = merge_type_chrom_data(base_dir, cancer_type, file_ext, start_col_name=start_col_name)
        out["TYPE_CENTERS"].setdefault(cancer_type, {})
        if include_debug:
            out["DEBUG_META"].setdefault(cancer_type, {})

        for chrom, lists in chrom_map.items():
            if chrom not in merged:
                print(f"[WARN] No data found for {cancer_type} {chrom}; keeping empty lists.")
                out["TYPE_CENTERS"][cancer_type][chrom] = {"pos": [], "neg": []}
                if include_debug:
                    out["DEBUG_META"][cancer_type][chrom] = {"pos": [], "neg": []}
                continue

            starts = merged[chrom]["starts"]
            pos_mb = lists.get("pos", []) or []
            neg_mb = lists.get("neg", []) or []

            pos_idx = []
            neg_idx = []
            pos_dbg = []
            neg_dbg = []

            for mb in pos_mb:
                try:
                    idx = mb_to_index(starts, mb)
                    pos_idx.append(int(idx))
                    if include_debug:
                        pos_dbg.append({
                            "orig_mb": float(mb),
                            "matched_bp": float(starts[idx])
                        })
                except Exception as e:
                    print(f"[WARN] pos conversion failed for {cancer_type} {chrom} {mb} -> {e}")

            for mb in neg_mb:
                try:
                    idx = mb_to_index(starts, mb)
                    neg_idx.append(int(idx))
                    if include_debug:
                        neg_dbg.append({
                            "orig_mb": float(mb),
                            "matched_bp": float(starts[idx])
                        })
                except Exception as e:
                    print(f"[WARN] neg conversion failed for {cancer_type} {chrom} {mb} -> {e}")

            out["TYPE_CENTERS"][cancer_type][chrom] = {"pos": pos_idx, "neg": neg_idx}
            if include_debug:
                out["DEBUG_META"][cancer_type][chrom] = {"pos": pos_dbg, "neg": neg_dbg}

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Convert TYPE_CENTERS (Mb coordinates) to row indices using dataset starts."
    )
    ap.add_argument("--config", "-c", type=str, default=None,
                    help="Path to global config.yaml (default: $GENSAMPLES_CONFIG or ./config.yaml)")
    ap.add_argument("--centers", "-t", type=str, default=None,
                    help="Path to type_centers.yaml (default: $GENSAMPLES_TYPE_CENTERS or ./type_centers.yaml)")
    ap.add_argument("--output", "-o", type=str, default="type_centers_index.yaml",
                    help="Output YAML file path for indexed centers.")
    ap.add_argument("--start-col-name", type=str, default=None,
                    help="Column name to use as genomic start (bp). Default: FIRST column in files.")
    ap.add_argument("--include-debug", action="store_true",
                    help="Include DEBUG_META (orig Mb and matched bp) for auditing.")
    args = ap.parse_args()

    global_cfg = load_global_config(args.config)
    centers_cfg = load_type_centers(args.centers)

    result = convert_centers_to_indices(
        global_cfg,
        centers_cfg,
        start_col_name=args.start_col_name,
        include_debug=args.include_debug
    )

    # Write YAML
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] Wrote indexed centers to: {args.output}")


if __name__ == "__main__":
    main()
