/*
最新版本-记录于2025.3.29

使用方法: ./gen_bin_del cancer_type
如: ./gen_bin_del BRCA

代码功能与逻辑：
生成缺失类型的bin
与扩增类型略有不同，只对cn<2的cnv样本进行统计，去除了扩增的cnv
出发点是：
由于扩增的范围是大于2的任意整数，而缺失只能是1和0
并且缺失的样本较少，混在一起会掩盖缺失的信号
在vis_hm.py以mix模式执行可以观察到

本代码处理逻辑：
先按染色体分组
第一步：先对所有 Copy_Number<2 的记录，提取其 Start 与 End 作为 bin 的边界点
第二步：仅对缺失（Copy_Number<2）记录进行累减，并严格要求区间包含整个bin：
    遍历该染色体的缺失记录列表
    如果记录的区间 [Start, End] 完全覆盖当前 bin (rec.Start <= bin.start && rec.End > bin.end)，
    则对应样本的 n 值减去 (2 - Copy_Number)
最终输出：每个染色体生成一个 cnv_{chromosome}.tsv，包含 bin 位置及各样本的缺失 n 值
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <filesystem>

// 定义记录结构体
struct Record {
    std::string GDC_Aliquot;
    std::string Chromosome;
    long Start;
    long End;
    int Copy_Number;
    int Major_Copy_Number;
    int Minor_Copy_Number;
};

// 定义 Bin 结构体
struct Bin {
    long start;
    long end;
    // 每个样本的当前 n 值
    std::map<std::string, int> n_values;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cancer_type>\n";
        return 1;
    }

    std::string cancer_type = argv[1];
    std::string input_file = "/Users/sanjati/jangoTemp/temp2/pycharmD/Data/merged_dataframe/merged_dataframe_" + cancer_type + ".tsv";

    std::ifstream infile(input_file);
    if (!infile) {
        std::cerr << "无法打开输入文件: " << input_file << "\n";
        return 1;
    }

    // 读取所有记录
    std::string header;
    std::getline(infile, header);

    std::vector<Record> records;
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        Record rec;
        std::string token;
        std::getline(iss, rec.GDC_Aliquot, '\t');
        std::getline(iss, rec.Chromosome, '\t');
        std::getline(iss, token, '\t'); rec.Start = std::stol(token);
        std::getline(iss, token, '\t'); rec.End = std::stol(token);
        std::getline(iss, token, '\t'); rec.Copy_Number = std::stoi(token);
        std::getline(iss, token, '\t'); rec.Major_Copy_Number = std::stoi(token);
        std::getline(iss, token, '\t'); rec.Minor_Copy_Number = std::stoi(token);
        records.push_back(rec);
    }
    infile.close();

    // 按染色体分组
    std::map<std::string, std::vector<Record>> groups;
    for (const auto &rec : records) {
        groups[rec.Chromosome].push_back(rec);
    }

    // 创建输出目录
    std::string base_output_dir = "/Users/sanjati/jangoTemp/temp2/pycharmD/preprocess/Version0209/output/bin_with_case_del/" + cancer_type;
    std::filesystem::create_directories(base_output_dir);

    for (auto &pair : groups) {
        const auto &chromosome = pair.first;
        auto &group_records = pair.second;

        // 第一步：提取 Copy_Number<2 的边界点
        std::set<long> points;
        for (const auto &rec : group_records) {
            if (rec.Copy_Number < 2) {
                points.insert(rec.Start);
                points.insert(rec.End);
            }
        }
        if (points.size() < 2) {
            std::cerr << "染色体 " << chromosome << " 边界点不足, 跳过\n";
            continue;
        }
        std::vector<long> sorted_points(points.begin(), points.end());

        // 收集样本列表
        std::set<std::string> aliquot_set;
        for (const auto &rec : group_records) aliquot_set.insert(rec.GDC_Aliquot);
        std::vector<std::string> aliquot_list(aliquot_set.begin(), aliquot_set.end());

        // 初始化 bins
        std::vector<Bin> bins;
        for (size_t i = 0; i < sorted_points.size()-1; ++i) {
            Bin b; b.start = sorted_points[i]; b.end = sorted_points[i+1];
            for (auto &a : aliquot_list) b.n_values[a] = 2;
            bins.push_back(b);
        }

        // 第二步：仅对缺失段累减 n 值，并避免边界双重计数
        for (auto &bin : bins) {
            for (const auto &rec : group_records) {
                if (rec.Copy_Number < 2 && rec.Start <= bin.start && rec.End > bin.end) {
                    int diff = 2 - rec.Copy_Number;  // 1 或 2
                    bin.n_values[rec.GDC_Aliquot] -= diff;
                }
            }
        }

        // 输出 TSV
        std::string out_file = base_output_dir + "/cnv_" + chromosome + ".tsv";
        std::ofstream ofs(out_file);
        ofs << "Chromosome\tStart\tEnd";
        for (auto &a : aliquot_list) ofs << "\t" << a;
        ofs << "\n";
        for (const auto &bin : bins) {
            ofs << chromosome << "\t" << bin.start << "\t" << bin.end;
            for (auto &a : aliquot_list) ofs << "\t" << bin.n_values.at(a);
            ofs << "\n";
        }
        ofs.close();
        std::cout << "染色体 " << chromosome << " 数据已写入 " << out_file << "\n";
    }

    return 0;
}
