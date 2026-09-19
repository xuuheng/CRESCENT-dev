#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import yaml
import pandas as pd
import numpy as np
import importlib.util
from typing import Optional, Dict, Any
from collections.abc import Sequence  # 用于判断序列类型


# ========== 可配置变量 ==========
GLOBAL_CONFIG_PATH = "./config.yaml"
INPUT_CENTERS_PY  = "./aaa.py"
OUTPUT_PATH       = "./baseline_o.yaml"
START_COL_NAME    = None
INCLUDE_DEBUG     = True
# 自动去重容差（近似比较时 |a-b| <= tol 视为重复）
DEDUP_TOL         = 1e-6
# =================================


def load_global_config(path: Optional[str]) -> Dict[str, Any]:
    print(f"[DEBUG] 尝试加载全局配置: {path}")
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"未找到全局配置文件: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"[DEBUG] 读取到的 config.yaml 内容: {cfg}")
    required = ["DATA_BASE_DIR", "FILE_EXTENSION"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"缺少必要键: {missing}")
    return cfg


# ====================== TYPE_CENTERS 归一化与去重 ======================

def _as_dict_type_centers(tc_raw):
    """
    把 TYPE_CENTERS 归一化为 dict。
    允许:
      - 直接是 dict
      - 是 tuple/list，里面包含一个 dict（例如末尾多逗号导致的 ({...},)）
    """
    if isinstance(tc_raw, dict):
        return tc_raw
    if isinstance(tc_raw, Sequence) and not isinstance(tc_raw, (str, bytes)):
        for item in tc_raw:
            if isinstance(item, dict):
                return item
    raise TypeError(
        f"TYPE_CENTERS 必须是 dict，或包含 dict 的 tuple/list；当前类型: {type(tc_raw).__name__}"
    )


def _dedup_numeric_list_keep_order(seq, tol=1e-6):
    """
    对包含数字(或可转成数字)的列表做去重，保留顺序。
    - 浮点数使用近似比较：|a-b| <= tol 认为重复
    - 不能转为 float 的元素使用严格相等比较
    """
    out = []
    seen = []  # 存放对比基准(尽量转为 float)
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
    """
    就地去重 TYPE_CENTERS 中各 chr 的 pos/neg。
    """
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
                print(f"[INFO] 去重 {st} {chrom}: "
                      f"pos {len(pos)} -> {len(pos_dedup)}, "
                      f"neg {len(neg)} -> {len(neg_dedup)}")

            pn["pos"] = pos_dedup
            pn["neg"] = neg_dedup
# =====================================================================


def load_type_centers_from_py(py_path: str) -> Dict[str, Any]:
    print(f"[DEBUG] 尝试加载输入文件: {py_path}")
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"未找到输入文件: {py_path!r}")

    spec = importlib.util.spec_from_file_location("centers_module", py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    if not hasattr(mod, "TYPE_CENTERS"):
        raise KeyError("输入文件未定义 TYPE_CENTERS")

    tc_raw = getattr(mod, "TYPE_CENTERS")
    print(f"[DEBUG] 读取到的 TYPE_CENTERS 原始类型: {type(tc_raw).__name__}")

    # 归一化: 支持 dict 或 (dict,) 等
    tc = _as_dict_type_centers(tc_raw)

    # 自动去重（可调 tol）
    _dedup_type_centers_inplace(tc, tol=DEDUP_TOL, verbose=True)

    print(f"[DEBUG] 归一化并去重后 TYPE_CENTERS keys: {list(tc.keys())}")
    return {"TYPE_CENTERS": tc}


def merge_type_chrom_data(base_dir: str,
                          sample_type: str,
                          file_ext: str,
                          start_col_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    type_dir = os.path.join(base_dir, sample_type)
    file_list = glob.glob(os.path.join(type_dir, f"*{file_ext}"))
    print(f"[DEBUG] {sample_type} 在 {type_dir} 找到 {len(file_list)} 个文件: {file_list}")

    chrom_frames = {}
    for fp in file_list:
        try:
            df = pd.read_csv(fp, sep="\t")
        except Exception as e:
            print(f"[WARN] 读取失败: {fp} -> {e}")
            continue

        print(f"[DEBUG] 读取 {fp} 成功, 列: {list(df.columns)}, 行数: {len(df)}")

        if "chr" not in df.columns:
            print(f"[WARN] 文件缺少 'chr' 列, 跳过: {fp}")
            continue

        if start_col_name and start_col_name in df.columns:
            start_series = pd.to_numeric(df[start_col_name], errors="coerce")
            print(f"[DEBUG] 使用列 {start_col_name} 作为 start")
        else:
            first_col = df.columns[0]
            start_series = pd.to_numeric(df[first_col], errors="coerce")
            print(f"[DEBUG] 使用第一列 {first_col} 作为 start")

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
            print(f"[WARN] {sample_type} {chrom} 的 start 均为 NaN, 跳过")
            continue
        df_chrom = df_chrom.loc[valid].reset_index(drop=True)
        starts = starts[valid]
        print(f"[DEBUG] {sample_type} {chrom} 合并后行数: {len(df_chrom)}")
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

    print(f"[DEBUG] 将处理 {len(type_centers)} 个类型: {list(type_centers.keys())}")

    for sample_type, chrom_map in type_centers.items():
        print(f"[INFO] 处理类型: {sample_type}")
        merged = merge_type_chrom_data(base_dir, sample_type, file_ext, start_col_name=start_col_name)
        out["TYPE_CENTERS"].setdefault(sample_type, {})
        if include_debug:
            out["DEBUG_META"].setdefault(sample_type, {})

        for chrom, lists in chrom_map.items():
            print(f"[DEBUG] 处理 {sample_type} {chrom}, pos 数量={len(lists.get('pos', []))}, neg 数量={len(lists.get('neg', []))}")
            if chrom not in merged:
                print(f"[WARN] 无数据: {sample_type} {chrom}, 输出空列表")
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


# ====================== 新增：打印统计 ======================

def report_type_centers_counts(type_centers: dict):
    """
    打印每个 cancer_type 的 pos/neg 数量和总数
    """
    print("\n=== 各癌种位点数量统计（去重后） ===")
    for cancer_type, chroms in type_centers.items():
        pos_count = sum(len(v.get("pos", [])) for v in chroms.values())
        neg_count = sum(len(v.get("neg", [])) for v in chroms.values())
        total = pos_count + neg_count
        print(f"{cancer_type:10s} -> pos: {pos_count:4d}, neg: {neg_count:4d}, total: {total:4d}")


# ==========================================================


def main():
    global_cfg = load_global_config(GLOBAL_CONFIG_PATH)
    centers_cfg = load_type_centers_from_py(INPUT_CENTERS_PY)

    # 新增：打印每个 cancer_type 的 pos/neg/总数（已去重后的数据）
    report_type_centers_counts(centers_cfg["TYPE_CENTERS"])

    result = convert_centers_to_indices(
        global_cfg,
        centers_cfg,
        start_col_name=START_COL_NAME,
        include_debug=INCLUDE_DEBUG
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] 已写出: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
