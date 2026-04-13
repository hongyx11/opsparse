#ifndef OPSPARSE_IO_MATRIX_MARKET_HPP_
#define OPSPARSE_IO_MATRIX_MARKET_HPP_

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace opsparse::io {

struct MatrixMarketBanner {
    std::string matrix;    // "matrix" or "vector"
    std::string storage;   // "array" or "coordinate"
    std::string type;      // "complex", "real", "integer", "pattern"
    std::string symmetry;  // "general", "symmetric", "hermitian", "skew-symmetric"
};

inline void tokenize(std::vector<std::string> &tokens, const std::string &str, const std::string &delimiters = "\n\r\t ")
{
    tokens.clear();
    auto first_pos = str.find_first_not_of(delimiters, 0);
    auto last_pos = str.find_first_of(delimiters, first_pos);
    while (std::string::npos != first_pos || std::string::npos != last_pos) {
        tokens.push_back(str.substr(first_pos, last_pos - first_pos));
        first_pos = str.find_first_not_of(delimiters, last_pos);
        last_pos = str.find_first_of(delimiters, first_pos);
    }
}

template <typename Stream>
inline void read_mm_banner(Stream &input, MatrixMarketBanner &banner)
{
    std::string line;
    std::vector<std::string> tokens;
    std::getline(input, line);
    tokenize(tokens, line);

    if (tokens.size() != 5 || tokens[0] != "%%MatrixMarket" || tokens[1] != "matrix")
        throw std::runtime_error("invalid MatrixMarket banner");

    banner.matrix = tokens[1];
    banner.storage = tokens[2];
    banner.type = tokens[3];
    banner.symmetry = tokens[4];

    if (banner.matrix != "matrix" && banner.matrix != "vector")
        throw std::runtime_error("invalid MatrixMarket matrix type: " + banner.matrix);
    if (banner.matrix == "vector") throw std::runtime_error("not impl matrix type: " + banner.matrix);

    if (banner.storage != "array" && banner.storage != "coordinate")
        throw std::runtime_error("invalid MatrixMarket storage format [" + banner.storage + "]");
    if (banner.storage == "array") throw std::runtime_error("not impl storage type " + banner.storage);

    if (banner.type != "complex" && banner.type != "real" && banner.type != "integer" && banner.type != "pattern")
        throw std::runtime_error("invalid MatrixMarket data type [" + banner.type + "]");

    // MatrixMarket spec allows mixed-case keywords; normalise to lowercase.
    for (auto &ch : banner.symmetry)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (banner.symmetry != "general" && banner.symmetry != "symmetric" && banner.symmetry != "hermitian" &&
        banner.symmetry != "skew-symmetric")
        throw std::runtime_error("invalid MatrixMarket symmetry [" + banner.symmetry + "]");
    if (banner.symmetry == "hermitian") throw std::runtime_error("not impl matrix type: " + banner.symmetry);
}

}  // namespace opsparse::io

#endif  // OPSPARSE_IO_MATRIX_MARKET_HPP_
