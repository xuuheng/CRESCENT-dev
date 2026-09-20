import os
import glob
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_intervals(file_path):
    """Load an interval file and add a chromosome value without the `chr` prefix."""
    df = pd.read_csv(file_path, sep='\t')
    if 'Chromosome' in df.columns:
        df['Chromosome_clean'] = df['Chromosome'].astype(str).str.replace('chr', '', regex=False)
    else:
        raise ValueError(f"Column 'Chromosome' was not found in {file_path}")
    return df


def process_file_A(fileA_path, df_B, df_C, output_dir):
    """Label intervals by complete containment in both reference sets."""
    df_A = pd.read_csv(fileA_path, sep='\t')

    if 'chr' in df_A.columns:
        df_A['chr_clean'] = df_A['chr'].astype(str).str.replace('chr', '', regex=False)
    else:
        raise ValueError(f"Column 'chr' was not found in {fileA_path}")

    def in_interval(row, df_intervals):
        ch = row['chr_clean']
        start = row['start']
        end = row['end']
        sub = df_intervals[df_intervals['Chromosome_clean'] == ch]
        if sub.empty:
            return 0
        contained = ((sub['Start'] <= start) & (sub['End'] >= end)).any()
        return int(contained)

    df_A['in_B'] = df_A.apply(lambda row: in_interval(row, df_B), axis=1)
    df_A['in_C'] = df_A.apply(lambda row: in_interval(row, df_C), axis=1)
    df_A['in_B_and_C'] = df_A.apply(lambda row: 1 if (row['in_B'] == 1 and row['in_C'] == 1) else 0, axis=1)

    df_A.drop(columns=['chr_clean'], inplace=True)

    output_file = os.path.join(output_dir, os.path.basename(fileA_path))
    df_A.to_csv(output_file, sep='\t', index=False)
    print(f"Processed: {fileA_path} -> {output_file}")


def main():
    cancer_type="BRCA"
    mut_type="amp"
    dir_A = PROJECT_ROOT / "Data" / "Matrix_cluster" / cancer_type
    GISTIC = PROJECT_ROOT / "Data" / "GISTIC_output_extracted" / cancer_type / "amp.tsv"
    RUBIC = PROJECT_ROOT / "Data" / "RUBIC_output" / "BRCA" / "gains" / f"{cancer_type}.tsv"
    output_dir = PROJECT_ROOT / "Data" / "labeled_2025" / cancer_type / mut_type

    os.makedirs(output_dir, exist_ok=True)

    df_B = load_intervals(RUBIC)
    df_C = load_intervals(GISTIC)

    fileA_list = glob.glob(os.path.join(dir_A, "*"))
    for fileA in fileA_list:
        process_file_A(fileA, df_B, df_C, output_dir)


if __name__ == "__main__":
    main()
