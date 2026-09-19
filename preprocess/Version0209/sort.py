#!/usr/bin/env python3
# 这个是最新版的总的臂级分类和划分bin二合一的代码
import pandas as pd
import os

def ope(cancer_type):
    # cancer_type="LUAD"
    # 参数设置（文件路径和容差比例）
    arm_file = "/Users/sanjati/jangoTemp/temp2/pycharmD/Data/ChromosomeData2025/GRCh38_Chromosome_Arm_Ranges.tsv"
    cnv_file = f"/Users/sanjati/jangoTemp/temp2/pycharmD/Data/merged_dataframe/merged_dataframe_{cancer_type}.tsv"
    output_dir = f"/Users/sanjati/jangoTemp/temp2/pycharmD/preprocess/Version0209/output/sorted/{cancer_type}"  # 输出文件夹
    k = 0.3  # 片段长度相对于臂长度的倍数（k倍，默认0.5）

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # -------------------------
    # 1. 读取染色体臂信息文件
    # 文件格式为 tab 分隔，前5列依次为：
    # chromosome, start0, end0, start1, end1
    # 分别对应 p臂和 q臂的范围（其中 q臂的终点即为染色体总长）
    arms_df = pd.read_csv(arm_file, sep="\t")

    # 重命名列，使其更直观
    arms_df.rename(columns={
        "start0": "p_arm_start",
        "end0": "p_arm_end",
        "start1": "q_arm_start",
        "end1": "q_arm_end"
    }, inplace=True)

    # 计算 p臂和 q臂的长度
    arms_df["p_arm_length"] = arms_df["p_arm_end"] - arms_df["p_arm_start"]
    arms_df["q_arm_length"] = arms_df["q_arm_end"] - arms_df["q_arm_start"]

    # 构造一个以染色体号为 key 的字典，便于查找
    arm_dict = {}
    for idx, row in arms_df.iterrows():
        chrom = row["chromosome"]
        arm_dict[chrom] = {
            "p": {"start": row["p_arm_start"], "end": row["p_arm_end"], "length": row["p_arm_length"]},
            "q": {"start": row["q_arm_start"], "end": row["q_arm_end"], "length": row["q_arm_length"]}
        }

    # -------------------------
    # 2. 读取CNV片段文件
    cnv_df = pd.read_csv(cnv_file, sep="\t")

    # 去除不需要的列：Major_Copy_Number和Minor_Copy_Number
    for col in ["Major_Copy_Number", "Minor_Copy_Number"]:
        if col in cnv_df.columns:
            cnv_df.drop(columns=[col], inplace=True)

    # 增加新列：is_arm_level, segment_length, arm_range, arm_length
    cnv_df["is_arm_level"] = False
    cnv_df["segment_length"] = 0
    cnv_df["arm_range"] = "N/A"
    cnv_df["arm_length"] = "N/A"

    # -------------------------
    # 3. 针对每个CNV片段进行判断
    for idx, row in cnv_df.iterrows():
        chrom = row["Chromosome"]
        seg_start = float(row["Start"])
        seg_end = float(row["End"])
        segment_length = seg_end - seg_start

        # 默认初始值
        is_arm = False
        arm_range = "N/A"
        arm_length_val = "N/A"

        if chrom in arm_dict:
            fully_in = False  # 标记是否完全落在某个臂内
            intersect = False  # 标记是否与某个臂有交集（但不完全包含）
            for arm in ["p", "q"]:
                arm_info = arm_dict[chrom][arm]
                # 如果片段完全在该臂范围内
                if seg_start >= arm_info["start"] and seg_end <= arm_info["end"]:
                    fully_in = True
                    arm_range = f"{arm_info['start']} - {arm_info['end']}"
                    arm_length_val = arm_info["length"]
                    # 根据k判断是否达到arm-level要求
                    if segment_length >= k * arm_info["length"]:
                        is_arm = True
                    break  # 只记录完全落在一个臂内的情况
                # 如果与某臂有交集（即部分在臂内、部分不在臂内）
                elif (seg_end > arm_info["start"]) and (seg_start < arm_info["end"]):
                    intersect = True
            # 如果没有完全在某一臂内但与某一臂有交集，则认为该片段跨区域（包含着丝粒等），标记为 -1
            if not fully_in and intersect:
                is_arm = True
                arm_range = "-1"
                arm_length_val = "-1"

        # 保存计算结果到对应列
        cnv_df.at[idx, "segment_length"] = segment_length
        cnv_df.at[idx, "arm_range"] = arm_range
        cnv_df.at[idx, "arm_length"] = arm_length_val
        cnv_df.at[idx, "is_arm_level"] = is_arm

    # -------------------------
    # 4. 按染色体分组，将结果输出到不同文件中
    for chrom, group in cnv_df.groupby("Chromosome"):
        if chrom in ["chrX", "chrY"]:
            continue
        out_file = os.path.join(output_dir, f"cnv_{chrom}.txt")
        group.to_csv(out_file, sep="\t", index=False)


    print("处理完成，各染色体文件保存在目录：", output_dir)

    import subprocess

    # 可执行文件路径（根据实际情况修改路径）
    executable_path = "./gen_bin_del"

    # 调用可执行文件，并传入参数
    result = subprocess.run([executable_path, cancer_type], capture_output=True, text=True)

    # 输出执行结果
    print("标准输出：", result.stdout)
    print("错误输出：", result.stderr)

if __name__ == "__main__":
    cancer_types = ["ACC", "SARC", "LGG", "HNSC", "ESCA", "LUSC","LAML", "CESC", "CHOL", "UCEC", "KICH", "BRCA", "BLCA", "KIRC", "GBM", "KIRP", "LIHC","COAD", "LUAD", "DLBC"]
    for cancer_type in cancer_types:
        ope(cancer_type)