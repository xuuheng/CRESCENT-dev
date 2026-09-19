# -
# *- coding: utf-8 -*-
"""
保持原脚本功能与逻辑不变，仅将 I/O 改造成：
- 从外部 YAML 读取全局配置与 TYPE_CENTERS
- 兼容 VALUE_FORMAT: "mb"（默认，单位百万）或 "index"（按行索引）
- 启动时清空并重建输出目录
- 其它行为（提取逻辑、归一化、裁剪、列处理、写出格式）与原版一致
"""

import os
import glob
import shutil
import traceback
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# YAML 配置加载（仿照“FINAL”样例）
# =============================================================================
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
    """
    读取并归一化配置。
    - global_path: 主配置 config.yaml（全局参数）
    - type_centers_path: type_centers.yaml（癌种-染色体-中心位置）
    环境变量：
      - GENSAMPLES_CONFIG
      - GENSAMPLES_TYPE_CENTERS
    默认：
      - ./config.yaml
      - ./type_centers.yaml
    """
    # 1) 全局配置路径
    if global_path is None:
        global_path = os.environ.get("GENSAMPLES_CONFIG")
    if global_path is None:
        global_path = _first_existing(["./config_del.yaml"])
    if global_path is None:
        raise FileNotFoundError("No global config file found. Set $GENSAMPLES_CONFIG or provide ./config_del.yaml.")

    with open(global_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 2) 中心配置路径
    centers_value_format = str(cfg.get("VALUE_FORMAT", "index")).lower()  # 默认“mb”，保持与你现有脚本一致
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

    # 3) 必需键
    required = [
        "DATA_BASE_DIR",        # 数据根目录（子目录为各癌种）
        "OUTPUT_DIR",           # 输出目录
        "FILE_EXTENSION",       # 输入文件扩展名，如 ".tsv"
        "FEATURE_COL_START",    # 特征起始列索引（基于 pandas iloc）
        "FIXED_FEATURE_COLS",   # 固定特征列数
        "SCALE_INPUT_SHAPES",   # 多尺度 [(H, W), ...]
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config key(s): {missing}")

    # 4) 派生键与默认
    cfg["FEATURE_COL_END"] = int(cfg["FEATURE_COL_START"]) + int(cfg["FIXED_FEATURE_COLS"])
    cfg.setdefault("TYPE_CENTERS", {})
    cfg.setdefault("NORM_METHOD", "col")          # 与原版一致：'col' | 'matrix' | None
    cfg.setdefault("OUTLIER_THRESHOLD", 10.0)     # 与原版一致：None 或 浮点数
    cfg.setdefault("CENTERS_VALUE_FORMAT", centers_value_format)

    # 列名配置（原版使用 "Chromosome" / "Start" / "End"）
    cfg.setdefault("CHR_COL", "Chromosome")
    cfg.setdefault("START_COL", "Start")
    cfg.setdefault("END_COL", "End")

    # 是否在运行开始时清空输出目录（原版默认清空）
    cfg.setdefault("CLEAN_OUTPUT_DIR", True)

    return cfg


CONFIG = load_config()

# =============================================================================
# 输出目录准备（保持原版行为：清空再创建）
# =============================================================================
if CONFIG.get("CLEAN_OUTPUT_DIR", True) and os.path.exists(CONFIG["OUTPUT_DIR"]):
    shutil.rmtree(CONFIG["OUTPUT_DIR"])
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# =============================================================================
# 核心逻辑（保持与你当前脚本一致）
# =============================================================================

def extract_matrix_from_feature_data(
    feature_data: np.ndarray,
    center_idx: int,
    H: int,
    W: int,
    norm_method: Optional[str] = 'col',  # None, 'col', 'matrix'
    outlier_threshold: Optional[float] = 10.0
) -> np.ndarray:
    """
    与原版一致：
      - 行中心窗口提取；不足 H 时先整体插值到 2000 行
      - 边界位置：先取顶/底部 H 行，再通过 np.roll 使中心行位于中间
      - 列按均值降序排序
      - 列数 < W：用“去最大值后”的均值生成常数列补齐
      - 列数 > W：轮转合并到前 W 列并做均值
      - 可选 'col' 或 'matrix' 标准化与离群裁剪（先全局后窗口）
    """
    feature_data = feature_data.astype(float)

    # 1) 预标准化
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

    # 2) 离群裁剪
    if outlier_threshold is not None:
        feature_data = np.clip(feature_data, -outlier_threshold, outlier_threshold)

    # 3) 行数不足时：插值到 2000 行
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

    # 4) 行方向中心窗口 + 边界滚动
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

    # 5) 列按均值降序排序
    col_means = np.mean(mat, axis=0)
    sorted_idx = np.argsort(-col_means)
    mat = mat[:, sorted_idx]

    # 6) 列数对齐到 W
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
    """写出与原版一致的 TSV 结构。"""
    with open(filepath, "w") as f:
        f.write(f"label: {label}\n")
        for i, mat in enumerate(scale_matrices):
            H, W = mat.shape
            f.write(f"scale: {i}, shape: {H}x{W}\n")
            for row in mat:
                f.write("\t".join(map(str, row)) + "\n")
            f.write("\n")
    print(f"写入: {filepath}")


# =============================================================================
# 生成逻辑（仅 I/O 方式改变，算法与处理逻辑不变）
# =============================================================================

def _load_type_frames(type_dir: str,
                      file_ext: str,
                      chr_col: str,
                      start_col: str) -> Dict[str, pd.DataFrame]:
    """
    装载某个癌种目录下所有文件，按染色体合并为 {chrom: df}。
    与原版保持一致：使用列名 Chromosome / Start / End（可在 YAML 中改名）。
    """
    file_list = glob.glob(os.path.join(type_dir, "*" + file_ext))
    chrom_data: Dict[str, List[pd.DataFrame]] = {}

    for filepath in file_list:
        try:
            df = pd.read_csv(filepath, sep="\t")
        except Exception as e:
            print(f"读取文件 {filepath} 时出错：{e}")
            continue

        if chr_col not in df.columns:
            print(f"警告: 文件 {filepath} 缺少 '{chr_col}' 列，已跳过。")
            continue
        if start_col not in df.columns:
            print(f"警告: 文件 {filepath} 缺少 '{start_col}' 列，已跳过。")
            continue

        for chrom in df[chr_col].dropna().unique():
            chrom_data.setdefault(str(chrom), []).append(df[df[chr_col] == chrom])

    merged: Dict[str, pd.DataFrame] = {}
    for chrom, parts in chrom_data.items():
        merged[chrom] = pd.concat(parts, ignore_index=True)

        # 确保 start 数值化
        merged[chrom][start_col] = pd.to_numeric(merged[chrom][start_col], errors="coerce")

    return merged


def _resolve_center_indices(
    centers_cfg: Dict[str, Any],
    starts: np.ndarray,
    value_format: str
) -> List[Tuple[int, float, int]]:
    """
    将中心位置（mb 或 index）解析为行索引：
    返回列表 [(row_idx, center_M, center_bp), ...]
    - value_format == "mb": 根据 starts（bp）寻找最近行
    - value_format == "index": 直接作为行索引
    """
    records: List[Tuple[int, float, int]] = []
    nrows = len(starts)

    if value_format == "mb":
        # 传入的 centers 是 float（单位：百万），需要 * 1e6 与 starts 匹配
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
                print(f"跳过索引 {idx}：超出范围 [0, {nrows-1}]")
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

    # 预载缓存 { (type_name, chrom): (starts, feature_data) }
    data_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    for type_name, chrom_dict in CONFIG["TYPE_CENTERS"].items():
        print(f"处理 type: {type_name}")
        type_counters.setdefault(type_name, {"pos": 0, "neg": 0})

        type_dir = os.path.join(base_dir, type_name)
        chrom_frames = _load_type_frames(type_dir, file_ext, chr_col, start_col)

        # 装载 starts 与特征矩阵切片
        for chrom, df_chrom in chrom_frames.items():
            try:
                starts = df_chrom[start_col].values.astype(float)
                feature_data = df_chrom.iloc[:, feat_start:feat_end].astype(float).values
            except Exception as e:
                print(f"提取 {type_name} {chrom} 特征失败：{e}")
                continue
            data_cache[(type_name, chrom)] = (starts, feature_data)

        # 遍历该 type 的所有染色体中心
        for chrom, centers in chrom_dict.items():
            if (type_name, chrom) not in data_cache:
                print(f"跳过: {type_name} 无 {chrom} 数据")
                continue

            starts, feature_data = data_cache[(type_name, chrom)]

            # 解析正负中心为行索引
            pos_records = _resolve_center_indices(centers.get("pos"), starts, value_format)
            neg_records = _resolve_center_indices(centers.get("neg"), starts, value_format)

            # 计数（保持与原逻辑相同：按配置数量累计有效条目）
            type_counters[type_name]["pos"] += len(pos_records)
            type_counters[type_name]["neg"] += len(neg_records)
            total_pos += len(pos_records)
            total_neg += len(neg_records)

            # 写出正负样本
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
                            print(f"跳过 {type_name} {chrom} {'正' if label==1 else '负'}例 center={center_m:.6f}Mb：{e}")
                            traceback.print_exc()
                            ok = False
                            break
                        if mat.shape != (H, W):
                            print(f"尺寸不符：得到 {mat.shape}，期望 ({H}, {W})")
                            ok = False
                            break
                        scale_mats.append(mat)
                    if not ok:
                        continue

                    if is_output:
                        fname = f"{type_name}_{chrom}_{int(center_bp)}_{'pos' if label == 1 else 'neg'}.tsv"
                        write_sample_tsv(os.path.join(output_dir, fname), scale_mats, label)

        # 小结
        tp = type_counters[type_name]["pos"]
        tn = type_counters[type_name]["neg"]
        print(f"[SUMMARY] {type_name}: pos = {tp}, neg = {tn}, total = {tp + tn}")

    # 总结
    print(f"[TOTAL] pos = {total_pos}, neg = {total_neg}, total = {total_pos + total_neg}")

    return {
        **type_counters,
        "__TOTAL__": {"pos": total_pos, "neg": total_neg, "total": total_pos + total_neg}
    }


if __name__ == "__main__":
    generate_samples(is_output=True)
