#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace tlmp {

// Used outside timed regions. Separate invocations append to the same table;
// an incompatible existing table is rejected instead of silently corrupted.
class CsvOutput {
 public:
  CsvOutput(const std::string& path, const std::string& header, bool enabled)
      : header_(header) {
    if (!enabled) {
      return;
    }
    if (path.empty()) {
      throw std::invalid_argument("--output requires a nonempty file path");
    }
    const std::filesystem::path destination(path);
    const bool has_data = std::filesystem::exists(destination) &&
                          std::filesystem::file_size(destination) != 0U;
    if (has_data) {
      std::ifstream existing(destination, std::ios::binary);
      std::string first_line;
      if (!std::getline(existing, first_line) || first_line + '\n' != header_) {
        throw std::runtime_error("CSV header mismatch in " + path +
                                 "; use --output with a new CSV file");
      }
      existing.seekg(-1, std::ios::end);
      if (existing.get() != '\n') {
        throw std::runtime_error("CSV has an incomplete last line: " + path);
      }
    }
    if (destination.has_parent_path()) {
      std::filesystem::create_directories(destination.parent_path());
    }
    stream_.open(destination, std::ios::app | std::ios::binary);
    if (!stream_) {
      throw std::runtime_error("cannot open CSV output: " + path);
    }
    stream_.exceptions(std::ios::badbit | std::ios::failbit);
    needs_header_ = !has_data;
  }

  void append(const std::string& row) {
    if (!stream_.is_open()) {
      return;
    }
    if (needs_header_) {
      stream_ << header_;
      needs_header_ = false;
    }
    stream_ << row;
    stream_.flush();
  }

 private:
  std::string header_;
  std::ofstream stream_;
  bool needs_header_ = false;
};

}  // namespace tlmp
