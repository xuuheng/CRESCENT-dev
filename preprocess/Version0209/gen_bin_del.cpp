/*
Build deletion-specific bins from records with Copy_Number < 2.

Start and End coordinates from deletion records define bin boundaries. Each
sample begins at the diploid value 2; a deletion covering an entire bin lowers
that value by (2 - Copy_Number). One TSV is written per chromosome.
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

struct Record {
    std::string GDC_Aliquot;
    std::string Chromosome;
    long Start;
    long End;
    int Copy_Number;
    int Major_Copy_Number;
    int Minor_Copy_Number;
};

struct Bin {
    long start;
    long end;
    std::map<std::string, int> n_values;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cancer_type> [project_root]\n";
        return 1;
    }

    std::string cancer_type = argv[1];
    std::filesystem::path project_root = argc >= 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::current_path();
    std::string input_file = (project_root / "Data" / "merged_dataframe" /
        ("merged_dataframe_" + cancer_type + ".tsv")).string();

    std::ifstream infile(input_file);
    if (!infile) {
        std::cerr << "Cannot open input file: " << input_file << "\n";
        return 1;
    }

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

    std::map<std::string, std::vector<Record>> groups;
    for (const auto &rec : records) {
        groups[rec.Chromosome].push_back(rec);
    }

    std::string base_output_dir = (project_root / "preprocess" / "Version0209" /
        "output" / "bin_with_case_del" / cancer_type).string();
    std::filesystem::create_directories(base_output_dir);

    for (auto &pair : groups) {
        const auto &chromosome = pair.first;
        auto &group_records = pair.second;

        std::set<long> points;
        for (const auto &rec : group_records) {
            if (rec.Copy_Number < 2) {
                points.insert(rec.Start);
                points.insert(rec.End);
            }
        }
        if (points.size() < 2) {
            std::cerr << "Skipping " << chromosome << ": insufficient deletion boundaries\n";
            continue;
        }
        std::vector<long> sorted_points(points.begin(), points.end());

        std::set<std::string> aliquot_set;
        for (const auto &rec : group_records) aliquot_set.insert(rec.GDC_Aliquot);
        std::vector<std::string> aliquot_list(aliquot_set.begin(), aliquot_set.end());

        std::vector<Bin> bins;
        for (size_t i = 0; i < sorted_points.size()-1; ++i) {
            Bin b; b.start = sorted_points[i]; b.end = sorted_points[i+1];
            for (auto &a : aliquot_list) b.n_values[a] = 2;
            bins.push_back(b);
        }

        // Use a half-open right boundary to avoid double-counting adjacent bins.
        for (auto &bin : bins) {
            for (const auto &rec : group_records) {
                if (rec.Copy_Number < 2 && rec.Start <= bin.start && rec.End > bin.end) {
                    int diff = 2 - rec.Copy_Number;
                    bin.n_values[rec.GDC_Aliquot] -= diff;
                }
            }
        }

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
        std::cout << "Wrote " << chromosome << " data to " << out_file << "\n";
    }

    return 0;
}
