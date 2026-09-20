#!/usr/bin/env python3
"""Classify arm-level segments and split focal CNAs by chromosome."""
import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def ope(cancer_type):
    arm_file = PROJECT_ROOT / "Data" / "ChromosomeData2025" / "GRCh38_Chromosome_Arm_Ranges.tsv"
    cnv_file = PROJECT_ROOT / "Data" / "merged_dataframe" / f"merged_dataframe_{cancer_type}.tsv"
    output_dir = PROJECT_ROOT / "preprocess" / "Version0209" / "output" / "sorted" / cancer_type
    k = 0.3  # Minimum segment-to-arm length ratio for an arm-level call.

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # The first five columns define chromosome, p-arm range, and q-arm range.
    arms_df = pd.read_csv(arm_file, sep="\t")

    arms_df.rename(columns={
        "start0": "p_arm_start",
        "end0": "p_arm_end",
        "start1": "q_arm_start",
        "end1": "q_arm_end"
    }, inplace=True)

    arms_df["p_arm_length"] = arms_df["p_arm_end"] - arms_df["p_arm_start"]
    arms_df["q_arm_length"] = arms_df["q_arm_end"] - arms_df["q_arm_start"]

    arm_dict = {}
    for idx, row in arms_df.iterrows():
        chrom = row["chromosome"]
        arm_dict[chrom] = {
            "p": {"start": row["p_arm_start"], "end": row["p_arm_end"], "length": row["p_arm_length"]},
            "q": {"start": row["q_arm_start"], "end": row["q_arm_end"], "length": row["q_arm_length"]}
        }

    cnv_df = pd.read_csv(cnv_file, sep="\t")

    for col in ["Major_Copy_Number", "Minor_Copy_Number"]:
        if col in cnv_df.columns:
            cnv_df.drop(columns=[col], inplace=True)

    cnv_df["is_arm_level"] = False
    cnv_df["segment_length"] = 0
    cnv_df["arm_range"] = "N/A"
    cnv_df["arm_length"] = "N/A"

    for idx, row in cnv_df.iterrows():
        chrom = row["Chromosome"]
        seg_start = float(row["Start"])
        seg_end = float(row["End"])
        segment_length = seg_end - seg_start

        is_arm = False
        arm_range = "N/A"
        arm_length_val = "N/A"

        if chrom in arm_dict:
            fully_in = False
            intersect = False
            for arm in ["p", "q"]:
                arm_info = arm_dict[chrom][arm]
                if seg_start >= arm_info["start"] and seg_end <= arm_info["end"]:
                    fully_in = True
                    arm_range = f"{arm_info['start']} - {arm_info['end']}"
                    arm_length_val = arm_info["length"]
                    if segment_length >= k * arm_info["length"]:
                        is_arm = True
                    break
                elif (seg_end > arm_info["start"]) and (seg_start < arm_info["end"]):
                    intersect = True
            # Segments crossing arm boundaries, including centromeres, are arm-level.
            if not fully_in and intersect:
                is_arm = True
                arm_range = "-1"
                arm_length_val = "-1"

        cnv_df.at[idx, "segment_length"] = segment_length
        cnv_df.at[idx, "arm_range"] = arm_range
        cnv_df.at[idx, "arm_length"] = arm_length_val
        cnv_df.at[idx, "is_arm_level"] = is_arm

    for chrom, group in cnv_df.groupby("Chromosome"):
        if chrom in ["chrX", "chrY"]:
            continue
        out_file = os.path.join(output_dir, f"cnv_{chrom}.txt")
        group.to_csv(out_file, sep="\t", index=False)


    print("Finished. Per-chromosome files were written to:", output_dir)

    import subprocess

    executable_path = Path(__file__).resolve().parent / "gen_bin_del"

    result = subprocess.run(
        [str(executable_path), cancer_type, str(PROJECT_ROOT)],
        capture_output=True,
        text=True,
    )

    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

if __name__ == "__main__":
    cancer_types = ["ACC", "SARC", "LGG", "HNSC", "ESCA", "LUSC","LAML", "CESC", "CHOL", "UCEC", "KICH", "BRCA", "BLCA", "KIRC", "GBM", "KIRP", "LIHC","COAD", "LUAD", "DLBC"]
    for cancer_type in cancer_types:
        ope(cancer_type)
