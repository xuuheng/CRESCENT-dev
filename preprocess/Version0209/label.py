import os
import glob
import pandas as pd


def load_intervals(file_path):
    """
    读取文件 B 或 C，返回 DataFrame 并增加一个标准化染色体列 (Chromosome_clean)：
    将 'chr' 前缀去掉，方便后续比较。
    """
    df = pd.read_csv(file_path, sep='\t')
    if 'Chromosome' in df.columns:
        df['Chromosome_clean'] = df['Chromosome'].astype(str).str.replace('chr', '', regex=False)
    else:
        raise ValueError(f"文件 {file_path} 中未找到 'Chromosome' 列")
    return df


def process_file_A(fileA_path, df_B, df_C, output_dir):
    """
    对单个 A 文件进行处理：
    - 读取 A 文件，生成标准化的染色体列 (chr_clean)（去掉 'chr' 前缀）
    - 对每一行判断其区间是否完全包含在 df_B 中（in_B）和 df_C 中（in_C）
    - 新增一列 in_B_and_C 为 in_B 和 in_C 的交集（均为1则为1，否则为0）
    - 保存结果到输出目录中，文件名保持不变
    """
    # 读取 A 文件，假设文件为制表符分隔
    df_A = pd.read_csv(fileA_path, sep='\t')

    # 检查并生成标准化染色体列（去掉可能存在的 "chr" 前缀）
    if 'chr' in df_A.columns:
        df_A['chr_clean'] = df_A['chr'].astype(str).str.replace('chr', '', regex=False)
    else:
        raise ValueError(f"文件 {fileA_path} 中未找到 'chr' 列")

    # 定义判断是否被任一区间包含的函数
    def in_interval(row, df_intervals):
        ch = row['chr_clean']
        start = row['start']
        end = row['end']
        # 选取染色体匹配的区间
        sub = df_intervals[df_intervals['Chromosome_clean'] == ch]
        if sub.empty:
            return 0
        # 检查是否存在一个区间满足：区间的 Start <= row.start 且区间的 End >= row.end
        contained = ((sub['Start'] <= start) & (sub['End'] >= end)).any()
        return int(contained)

    # 对每一行判断在 B 文件中是否被包含
    df_A['in_B'] = df_A.apply(lambda row: in_interval(row, df_B), axis=1)
    # 对每一行判断在 C 文件中是否被包含
    df_A['in_C'] = df_A.apply(lambda row: in_interval(row, df_C), axis=1)
    # 交集：只有 in_B 和 in_C 都为 1 时，该列才为 1
    df_A['in_B_and_C'] = df_A.apply(lambda row: 1 if (row['in_B'] == 1 and row['in_C'] == 1) else 0, axis=1)

    # 可选：删除辅助列
    df_A.drop(columns=['chr_clean'], inplace=True)

    # 输出文件，保持原文件名到指定输出目录
    output_file = os.path.join(output_dir, os.path.basename(fileA_path))
    df_A.to_csv(output_file, sep='\t', index=False)
    print(f"处理完成：{fileA_path} -> {output_file}")


def main():
    cancer_type="BRCA"
    mut_type="amp"  # 待扩展
    dir_A = f"/Users/sanjati/jangoTemp/temp2/pycharmD/Data/Matrix_cluster/{cancer_type}"  # 存储多个 A 类型文件的目录
    GISTIC = f"/Users/sanjati/jangoTemp/temp2/pycharmD/Data/GISTIC_output_extracted/{cancer_type}/amp.tsv"  # 文件 B 的路径
    RUBIC = f"/Users/sanjati/jangoTemp/temp2/pycharmD/Data/RUBIC_output/BRCA/gains/{cancer_type}.tsv"  # 文件 C 的路径
    output_dir = f"/Users/sanjati/jangoTemp/temp2/pycharmD/Data/labeled_2025/{cancer_type}/{mut_type}"  # 输出目录

    # 如果输出目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)

    # 读取文件 B 和文件 C
    df_B = load_intervals(RUBIC)
    df_C = load_intervals(GISTIC)

    # 获取目录中所有 A 文件（假设为所有文件，可根据需要修改匹配模式，如 "*.txt"）
    fileA_list = glob.glob(os.path.join(dir_A, "*"))
    for fileA in fileA_list:
        process_file_A(fileA, df_B, df_C, output_dir)


if __name__ == "__main__":
    main()
