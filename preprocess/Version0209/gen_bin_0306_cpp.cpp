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

struct Record {
    std::string GDC_Aliquot;
    std::string Chromosome;
    long long Start;
    long long End;
    double Copy_Number;
    bool is_arm_level;
};

struct BinResult {
    long long start;
    long long end;
    std::string chr;
    std::unordered_map<std::string, double> aliquotValues;
    double sum;
};

class Processor {
public:
    Processor(const std::string& inputDir, const std::string& outputDir)
        : input_dir(inputDir), output_dir(outputDir) {}

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
        std::vector<std::string> required = {"GDC_Aliquot", "Chromosome", "Start", "End", "Copy_Number", "is_arm_level"};
        std::unordered_map<std::string, int> colIndex;
        for (size_t i = 0; i < headers.size(); ++i) {
            colIndex[headers[i]] = static_cast<int>(i);
        }
        for (const auto& col : required) {
            if (colIndex.find(col) == colIndex.end()) {
                std::cerr << "Error in file " << file_path << ": missing required column: " << col << std::endl;
                return {};
            }
        }

        std::vector<Record> records;
        std::string line;
        while (std::getline(infile, line)) {
            if(line.empty()) continue;
            std::vector<std::string> tokens = split(line, '\t');
            if(tokens.size() < headers.size()) continue;
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
            rec.is_arm_level = (isArmStr == "True" || isArmStr == "true");
            records.push_back(rec);
        }
        infile.close();

        std::vector<Record> filtered;
        for (const auto& rec : records) {
            if (!rec.is_arm_level)
                filtered.push_back(rec);
        }

        std::set<long long> pointSet;
        for (const auto& rec : filtered) {
            pointSet.insert(rec.Start);
            pointSet.insert(rec.End);
        }
        if (pointSet.size() < 2) {
            std::cerr << "Warning: insufficient Start and End coordinates to create bins in " << file_path << std::endl;
            return {};
        }
        std::vector<long long> points(pointSet.begin(), pointSet.end());
        std::sort(points.begin(), points.end());

        std::vector<std::pair<long long, long long>> bins;
        for (size_t i = 0; i < points.size() - 1; ++i) {
            bins.emplace_back(points[i], points[i+1]);
        }

        std::vector<BinResult> results;
        std::unordered_map<std::string, double> global_cn_totals;

        std::string chrValue = "";
        if (!filtered.empty()) {
            chrValue = filtered[0].Chromosome;
        }

        for (const auto& b : bins) {
            BinResult binRes;
            binRes.start = b.first;
            binRes.end = b.second;
            binRes.chr = chrValue;
            binRes.sum = 0.0;
            for (const auto& rec : filtered) {
                if (rec.Start <= b.first && rec.End >= b.second) {
                    binRes.aliquotValues[rec.GDC_Aliquot] += rec.Copy_Number;
                    global_cn_totals[rec.GDC_Aliquot] += rec.Copy_Number;
                }
            }
            double sum = 0.0;
            for (const auto& p : binRes.aliquotValues) {
                sum += p.second;
            }
            binRes.sum = sum;
            results.push_back(binRes);
        }

        std::vector<std::pair<std::string, double>> aliquotVec(global_cn_totals.begin(), global_cn_totals.end());
        std::sort(aliquotVec.begin(), aliquotVec.end(), [](auto& a, auto& b) {
            return a.second > b.second;
        });
        std::vector<std::string> aliquotIds;
        for (const auto& p : aliquotVec) {
            aliquotIds.push_back(p.first);
        }

        for (auto& binRes : results) {
            for (const auto& id : aliquotIds) {
                if (binRes.aliquotValues.find(id) == binRes.aliquotValues.end()) {
                    binRes.aliquotValues[id] = 0.0;
                }
            }
        }

        std::sort(results.begin(), results.end(), [](const BinResult& a, const BinResult& b) {
            return a.start < b.start;
        });

        sorted_aliquotIds = aliquotIds;
        return results;
    }

    void process_all_files() {
        if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
            std::cerr << "Error: input directory does not exist or is not a directory: " << input_dir << std::endl;
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
            fs::path output_file = fs::path(output_dir) / file_path.filename();
            std::ofstream outfile(output_file);
            if (!outfile.is_open()) {
                std::cerr << "Error writing output for file " << file_path << std::endl;
                continue;
            }
            outfile << "start\tend";
            if (!results.empty() && !results[0].chr.empty()) {
                outfile << "\tchr";
            }
            for (const auto& id : sorted_aliquotIds) {
                outfile << "\t" << id;
            }
            outfile << "\tsum\n";

            // Represent uncovered sample bins with the diploid value 2.
            for (const auto& binRes : results) {
                outfile << binRes.start << "\t" << binRes.end;
                if (!binRes.chr.empty()) {
                    outfile << "\t" << binRes.chr;
                }
                for (const auto& id : sorted_aliquotIds) {
                    double value = binRes.aliquotValues.at(id);
                    if (value == 0.0)
                        outfile << "\t2";
                    else
                        outfile << "\t" << value;
                }
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
    std::vector<std::string> sorted_aliquotIds;

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
    std::string input_dir = (project_root / "preprocess" / "Version0209" / "output" / "sorted" / cancer_type).string();
    std::string output_dir = (project_root / "preprocess" / "Version0209" / "output" / "bin_with_case" / cancer_type).string();

    Processor processor(input_dir, output_dir);
    processor.process_all_files();
    return 0;
}
