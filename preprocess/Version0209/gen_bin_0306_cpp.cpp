#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;

// 定义一个结构体用于存储 TSV 文件中的每一行数据
struct Record {
    std::string GDC_Aliquot;
    std::string Chromosome;
    long long Start;
    long long End;
    double Copy_Number;
    bool is_arm_level;
};

// 定义一个结构体用于存储 bin 的结果
struct BinResult {
    long long start;
    long long end;
    std::string chr;  // 若文件中有 Chromosome 列，则使用第1行的染色体标识
    std::unordered_map<std::string, double> aliquotValues; // 每个 GDC_Aliquot 对应的累加值
    double sum;
};

class Processor {
public:
    Processor(const std::string& inputDir, const std::string& outputDir)
        : input_dir(inputDir), output_dir(outputDir) {}

    // 处理单个 TSV 文件，返回处理后的 bin 结果（若发生错误则返回空 vector）
    std::vector<BinResult> process_file(const fs::path& file_path) {
        std::ifstream infile(file_path);
        if (!infile.is_open()) {
            std::cerr << "Error reading file " << file_path << std::endl;
            return {};
        }

        std::string headerLine;
        if (!std::getline(infile, headerLine)) {
            std::cerr << "Error reading header from " << file_path << std::endl;
            return {};
        }
        std::vector<std::string> headers = split(headerLine, '\t');
        // 检查必须的列
        std::vector<std::string> required = {"GDC_Aliquot", "Chromosome", "Start", "End", "Copy_Number", "is_arm_level"};
        std::unordered_map<std::string, int> colIndex;
        for (size_t i = 0; i < headers.size(); ++i) {
            colIndex[headers[i]] = static_cast<int>(i);
        }
        for (const auto& col : required) {
            if (colIndex.find(col) == colIndex.end()) {
                std::cerr << "Error in file " << file_path << ": 缺少必要的列: " << col << std::endl;
                return {};
            }
        }

        // 读取所有行数据
        std::vector<Record> records;
        std::string line;
        while (std::getline(infile, line)) {
            if(line.empty()) continue;
            std::vector<std::string> tokens = split(line, '\t');
            if(tokens.size() < headers.size()) continue; // 跳过格式错误的行
            Record rec;
            rec.GDC_Aliquot = tokens[colIndex["GDC_Aliquot"]];
            rec.Chromosome = tokens[colIndex["Chromosome"]];
            try {
                rec.Start = std::stoll(tokens[colIndex["Start"]]);
                rec.End = std::stoll(tokens[colIndex["End"]]);
                rec.Copy_Number = std::stod(tokens[colIndex["Copy_Number"]]);
            } catch(const std::exception& e) {
                std::cerr << "Error converting numeric values in file " << file_path << ": " << e.what() << std::endl;
                continue;
            }
            std::string isArmStr = tokens[colIndex["is_arm_level"]];
            // 假设 is_arm_level 字段为 "True" 或 "False"
            rec.is_arm_level = (isArmStr == "True" || isArmStr == "true");
            records.push_back(rec);
        }
        infile.close();

        // 过滤掉 is_arm_level 为 True 的行
        std::vector<Record> filtered;
        for (const auto& rec : records) {
            if (!rec.is_arm_level)
                filtered.push_back(rec);
        }

        // 获取所有 Start 与 End 坐标，去重后排序
        std::set<long long> pointSet;
        for (const auto& rec : filtered) {
            pointSet.insert(rec.Start);
            pointSet.insert(rec.End);
        }
        if (pointSet.size() < 2) {
            std::cerr << "Warning: 文件 " << file_path << " 中的 Start 和 End 坐标不足以生成 bin" << std::endl;
            return {};
        }
        std::vector<long long> points(pointSet.begin(), pointSet.end());
        std::sort(points.begin(), points.end());

        // 生成相邻两个点构成的 bin 区间
        std::vector<std::pair<long long, long long>> bins;
        for (size_t i = 0; i < points.size() - 1; ++i) {
            bins.emplace_back(points[i], points[i+1]);
        }

        std::vector<BinResult> results;
        // 用于统计当前文件内各个 GDC_Aliquot 的全局累计 Copy_Number
        std::unordered_map<std::string, double> global_cn_totals;

        // 若存在 Chromosome 列，则取第一个过滤后记录的值
        std::string chrValue = "";
        if (!filtered.empty()) {
            chrValue = filtered[0].Chromosome;
        }

        // 遍历每个 bin
        for (const auto& b : bins) {
            BinResult binRes;
            binRes.start = b.first;
            binRes.end = b.second;
            binRes.chr = chrValue;
            binRes.sum = 0.0;
            // 遍历每个分段
            for (const auto& rec : filtered) {
                // 如果当前分段覆盖了该 bin（包含左右端点）
                if (rec.Start <= b.first && rec.End >= b.second) {
                    binRes.aliquotValues[rec.GDC_Aliquot] += rec.Copy_Number;
                    global_cn_totals[rec.GDC_Aliquot] += rec.Copy_Number;
                }
            }
            // 计算当前 bin 的总和
            double sum = 0.0;
            for (const auto& p : binRes.aliquotValues) {
                sum += p.second;
            }
            binRes.sum = sum;
            results.push_back(binRes);
        }

        // 对全局各 GDC_Aliquot 按累计总和降序排序
        std::vector<std::pair<std::string, double>> aliquotVec(global_cn_totals.begin(), global_cn_totals.end());
        std::sort(aliquotVec.begin(), aliquotVec.end(), [](auto& a, auto& b) {
            return a.second > b.second;
        });
        // 保存排序后的 aliquot id 顺序
        std::vector<std::string> aliquotIds;
        for (const auto& p : aliquotVec) {
            aliquotIds.push_back(p.first);
        }

        // 对每个 bin，补全缺失的 aliquot 列（填 0）并调整顺序（这里输出时按 aliquotIds 顺序输出）
        for (auto& binRes : results) {
            for (const auto& id : aliquotIds) {
                if (binRes.aliquotValues.find(id) == binRes.aliquotValues.end()) {
                    binRes.aliquotValues[id] = 0.0;
                }
            }
        }

        // 输出时将结果按照 bin 的起点排序
        std::sort(results.begin(), results.end(), [](const BinResult& a, const BinResult& b) {
            return a.start < b.start;
        });

        // 同时将排序后的 aliquot id 顺序保存到成员变量，便于写入输出文件时使用
        sorted_aliquotIds = aliquotIds;
        return results;
    }

    // 处理输入目录下所有文件，并将结果写入输出目录
    void process_all_files() {
        if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
            std::cerr << "Error: 输入目录 " << input_dir << " 不存在或不是一个目录" << std::endl;
            return;
        }
        fs::create_directories(output_dir);

        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (!entry.is_regular_file()) continue;
            fs::path file_path = entry.path();
            std::cout << "Processing file: " << file_path << std::endl;
            auto results = process_file(file_path);
            if (results.empty()) {
                std::cerr << "Skipping file " << file_path << " due to errors." << std::endl;
                continue;
            }
            // 写入输出文件
            fs::path output_file = fs::path(output_dir) / file_path.filename();
            std::ofstream outfile(output_file);
            if (!outfile.is_open()) {
                std::cerr << "Error writing output for file " << file_path << std::endl;
                continue;
            }
            // 写入表头：start, end, (chr), 各个 GDC_Aliquot 列，再 sum
            outfile << "start\tend";
            if (!results.empty() && !results[0].chr.empty()) {
                outfile << "\tchr";
            }
            for (const auto& id : sorted_aliquotIds) {
                outfile << "\t" << id;
            }
            outfile << "\tsum\n";

            // 写入每一行结果，增加逻辑将从第4列开始的 0 替换为 2
            for (const auto& binRes : results) {
                outfile << binRes.start << "\t" << binRes.end;
                if (!binRes.chr.empty()) {
                    outfile << "\t" << binRes.chr;
                }
                for (const auto& id : sorted_aliquotIds) {
                    double value = binRes.aliquotValues.at(id);
                    // 如果该值为 0，则替换为 2
                    if (value == 0.0)
                        outfile << "\t2";
                    else
                        outfile << "\t" << value;
                }
                // 检查 sum 列
                if (binRes.sum == 0.0)
                    outfile << "\t2\n";
                else
                    outfile << "\t" << binRes.sum << "\n";
            }
            outfile.close();
            std::cout << "Processed " << file_path << " -> " << output_file << std::endl;
        }
    }

private:
    std::string input_dir;
    std::string output_dir;
    // 保存全局排序后的 aliquot id 顺序，用于输出时的列顺序
    std::vector<std::string> sorted_aliquotIds;

    // 简单的字符串分割函数
    std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> elems;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            elems.push_back(item);
        }
        return elems;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cancer_type> [project_root]" << std::endl;
        return 1;
    }
    std::string cancer_type = argv[1];
    fs::path project_root = argc >= 3 ? fs::path(argv[2]) : fs::current_path();
    // 根据输入的 cancer_type 构造输入与输出目录
    std::string input_dir = (project_root / "preprocess" / "Version0209" / "output" / "sorted" / cancer_type).string();
    std::string output_dir = (project_root / "preprocess" / "Version0209" / "output" / "bin_with_case" / cancer_type).string();

    Processor processor(input_dir, output_dir);
    processor.process_all_files();
    return 0;
}
