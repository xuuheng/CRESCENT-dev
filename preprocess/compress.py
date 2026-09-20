import csv

import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mut="del"
host="se"
# Pipeline variant selection.
if mut=="amp":
    if host=="local":
        BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, 'bin_with_case')
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'bin_with_case_amp_compressed')
    else:
        BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, 'bin_with_case')
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'bin_with_case_amp_compressed_80')
elif mut=="del":
    if host=="local":
        BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, 'bin_with_case_del')
        OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'bin_with_case_del_compressed_80')
    else:
        BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, 'bin_with_case_del')
        OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'bin_with_case_del_compressed_40')
K = 40

def detect_sep(file_path: str) -> str:
    """Detect the delimiter from the first 2048 bytes, defaulting to tabs."""
    with open(file_path, 'r', newline='') as f:
        sample = f.read(2048)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters='\t,; ')
        return dialect.delimiter
    except csv.Error:
        return '\t'


def process_file(file_path: str, output_path: str, k: int, sep: str = '\t') -> None:
    """Compress sample columns to ``k`` while preserving metadata and suffix.

    The first three columns are metadata and the last column is a suffix.
    Missing feature columns are padded with diploid value 2. Remaining columns
    are assigned round-robin to the highest-mean anchors and averaged.
    """
    df = pd.read_csv(file_path, sep=sep)

    meta_cols   = df.columns[:3].tolist()
    suffix_cols = df.columns[-1:].tolist()

    data_cols = df.columns[3:-1].tolist()

    if len(data_cols) < k:
        num_pad = k - len(data_cols)
        pad_cols = [f'pad_{i+1}' for i in range(num_pad)]
        for col in pad_cols:
            df[col] = 2
        data_cols.extend(pad_cols)

    means    = df[data_cols].mean()
    top_cols = means.nlargest(k).index.tolist()

    rem_cols    = [c for c in data_cols if c not in top_cols]
    assignments = {col: top_cols[i % k] for i, col in enumerate(rem_cols)}

    merged_cols = {}
    for top in top_cols:
        group = [top] + [c for c, tgt in assignments.items() if tgt == top]
        merged_cols[top] = df[group].sum(axis=1) / len(group)

    df_out = pd.concat([
        df[meta_cols],
        pd.DataFrame(merged_cols),
        df[suffix_cols]
    ], axis=1)
    df_out = df_out[meta_cols + top_cols + suffix_cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, sep='\t', index=False)


def main():
    VALID_EXTS = ('.txt', '.tsv')

    for root, dirs, files in os.walk(BASE_INPUT_DIR):
        if root == BASE_INPUT_DIR:
            continue

        rel_dir = os.path.relpath(root, BASE_INPUT_DIR)
        out_dir = os.path.join(OUTPUT_DIR, rel_dir)

        for fname in files:
            if not fname.lower().endswith(VALID_EXTS):
                continue

            in_path  = os.path.join(root, fname)
            out_path = os.path.join(out_dir, fname)

            sep = detect_sep(in_path)
            process_file(in_path, out_path, K, sep=sep)
            print(f"Processed {in_path} (detected sep={repr(sep)}) -> {out_path}")


if __name__ == '__main__':
    main()
