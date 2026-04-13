#include "opsparse/core_ds/csr.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "opsparse/io/matrix_market.hpp"
#include "opsparse/system/cuda_common.hpp"

namespace opsparse {

namespace {

template <typename Index, typename Value>
struct Pair {
    Index ind;
    Value val;
    friend bool operator<=(const Pair &lhs, const Pair &rhs) { return lhs.ind <= rhs.ind; }
    friend bool operator<(const Pair &lhs, const Pair &rhs) { return lhs.ind < rhs.ind; }
    friend bool operator>(const Pair &lhs, const Pair &rhs) { return lhs.ind > rhs.ind; }
};

}  // namespace

void CSR::hrelease()
{
    delete[] rpt;
    rpt = nullptr;
    delete[] col;
    col = nullptr;
    delete[] val;
    val = nullptr;
}

void CSR::drelease()
{
    OPSPARSE_CHECK_CUDA(cudaFree(d_rpt));
    d_rpt = nullptr;
    OPSPARSE_CHECK_CUDA(cudaFree(d_col));
    OPSPARSE_CHECK_CUDA(cudaFree(d_val));
    d_col = nullptr;
    d_val = nullptr;
}

void CSR::release()
{
    hrelease();
    drelease();
}

CSR::~CSR() { release(); }

void CSR::H2D()
{
    drelease();
    OPSPARSE_CHECK_CUDA(cudaMalloc(&d_rpt, (M + 1) * sizeof(mint)));
    OPSPARSE_CHECK_CUDA(cudaMalloc(&d_col, nnz * sizeof(mint)));
    OPSPARSE_CHECK_CUDA(cudaMalloc(&d_val, nnz * sizeof(mdouble)));
    OPSPARSE_CHECK_CUDA(cudaMemcpy(d_rpt, rpt, (M + 1) * sizeof(mint), cudaMemcpyHostToDevice));
    OPSPARSE_CHECK_CUDA(cudaMemcpy(d_col, col, nnz * sizeof(mint), cudaMemcpyHostToDevice));
    OPSPARSE_CHECK_CUDA(cudaMemcpy(d_val, val, nnz * sizeof(mdouble), cudaMemcpyHostToDevice));
}

void CSR::D2H()
{
    hrelease();
    rpt = new mint[M + 1];
    col = new mint[nnz];
    val = new mdouble[nnz];
    OPSPARSE_CHECK_CUDA(cudaMemcpy(rpt, d_rpt, (M + 1) * sizeof(mint), cudaMemcpyDeviceToHost));
    OPSPARSE_CHECK_CUDA(cudaMemcpy(col, d_col, nnz * sizeof(mint), cudaMemcpyDeviceToHost));
    OPSPARSE_CHECK_CUDA(cudaMemcpy(val, d_val, nnz * sizeof(mdouble), cudaMemcpyDeviceToHost));
}

CSR::CSR(const CSR &A)
{
    M = A.M;
    N = A.N;
    nnz = A.nnz;
    rpt = new mint[M + 1];
    col = new mint[nnz];
    val = new mdouble[nnz];
    std::memcpy(rpt, A.rpt, (M + 1) * sizeof(mint));
    std::memcpy(col, A.col, nnz * sizeof(mint));
    std::memcpy(val, A.val, nnz * sizeof(mdouble));
}

CSR &CSR::operator=(const CSR &A)
{
    if (this == &A) return *this;
    hrelease();
    drelease();
    M = A.M;
    N = A.N;
    nnz = A.nnz;
    rpt = new mint[M + 1];
    col = new mint[nnz];
    val = new mdouble[nnz];
    std::memcpy(rpt, A.rpt, (M + 1) * sizeof(mint));
    std::memcpy(col, A.col, nnz * sizeof(mint));
    std::memcpy(val, A.val, nnz * sizeof(mdouble));
    return *this;
}

CSR::CSR(const CSR &A, mint M_, mint N_, mint M_start, mint N_start)
{
    assert(M_ + M_start <= A.M && "matrix subsect error M");
    assert(N_ + N_start <= A.N && "matrix subsect error N");
    const mint M_end = M_start + M_;
    const mint N_end = N_start + N_;
    M = M_;
    N = N_;
    mint *row_size = new mint[M];
    std::memset(row_size, 0, M * sizeof(mint));
    for (mint i = M_start; i < M_end; i++) {
        for (mint j = A.rpt[i]; j < A.rpt[i + 1]; j++) {
            if (A.col[j] >= N_start && A.col[j] < N_end) {
                row_size[i - M_start]++;
            }
        }
    }

    rpt = new mint[M + 1];
    rpt[0] = 0;
    for (mint i = 0; i < M; i++) {
        rpt[i + 1] = rpt[i] + row_size[i];
    }
    nnz = rpt[M];
    delete[] row_size;

    col = new mint[nnz];
    val = new mdouble[nnz];
    for (mint i = M_start; i < M_end; i++) {
        mint jj = rpt[i - M_start];
        for (mint j = A.rpt[i]; j < A.rpt[i + 1]; j++) {
            if (A.col[j] >= N_start && A.col[j] < N_end) {
                col[jj] = A.col[j] - N_start;
                val[jj++] = A.val[j];
            }
        }
    }
}

bool CSR::operator==(const CSR &rhs)
{
    if (nnz != rhs.nnz) {
        std::printf("nnz not equal %d %d\n", nnz, rhs.nnz);
        throw std::runtime_error("nnz not equal");
    }
    assert(M == rhs.M && "dimension not same");
    assert(N == rhs.N && "dimension not same");
    int error_num = 0;
    const double epsilon = 1e-9;
    for (mint i = 0; i < M; i++) {
        if (expect_false(error_num > 10)) throw std::runtime_error("matrix compare: error num exceed threshold");
        if (expect_false(rpt[i] != rhs.rpt[i])) {
            std::printf("rpt not equal at %d rows, %d != %d\n", i, rpt[i], rhs.rpt[i]);
            error_num++;
        }
        for (mint j = rpt[i]; j < rpt[i + 1]; j++) {
            if (expect_false(error_num > 10)) throw std::runtime_error("matrix compare: error num exceed threshold");
            if (col[j] != rhs.col[j]) {
                std::printf("col not equal at %d rows, index %d != %d\n", i, col[j], rhs.col[j]);
                error_num++;
            }
            if (!(std::fabs(val[j] - rhs.val[j]) < epsilon ||
                  std::fabs(val[j] - rhs.val[j]) < epsilon * std::fabs(val[j]))) {
                std::printf("val not eqaul at %d rows, value %.18le != %.18le\n", i, val[j], rhs.val[j]);
                error_num++;
            }
        }
    }
    if (rpt[M] != rhs.rpt[M]) {
        std::printf("rpt[M] not equal\n");
        throw std::runtime_error("matrix compare: error num exceed threshold");
    }
    return error_num == 0;
}

CSR::CSR(const std::string &mtx_file) { construct(mtx_file); }

void CSR::construct(const std::string &mtx_file)
{
    std::ifstream ifile(mtx_file.c_str());
    if (!ifile) {
        throw std::runtime_error(std::string("unable to open file \"") + mtx_file + std::string("\" for reading"));
    }
    io::MatrixMarketBanner banner;
    io::read_mm_banner(ifile, banner);

    std::string line;
    do {
        std::getline(ifile, line);
    } while (line[0] == '%');

    std::vector<std::string> tokens;
    io::tokenize(tokens, line);

    if (tokens.size() != 3) throw std::runtime_error("invalid MatrixMarket coordinate format");

    std::istringstream(tokens[0]) >> M;
    std::istringstream(tokens[1]) >> N;
    std::istringstream(tokens[2]) >> nnz;
    assert(nnz > 0 && "something wrong: nnz is 0");

    mint *I_ = new mint[nnz];
    mint *J_ = new mint[nnz];
    mdouble *coo_values_ = new mdouble[nnz];

    mint num_entries_read = 0;
    if (banner.type == "pattern") {
        while (num_entries_read < nnz && !ifile.eof()) {
            ifile >> I_[num_entries_read];
            ifile >> J_[num_entries_read];
            num_entries_read++;
        }
        std::fill(coo_values_, coo_values_ + nnz, mdouble(1));
    } else if (banner.type == "real" || banner.type == "integer") {
        while (num_entries_read < nnz && !ifile.eof()) {
            ifile >> I_[num_entries_read];
            ifile >> J_[num_entries_read];
            ifile >> coo_values_[num_entries_read];
            num_entries_read++;
        }
    } else if (banner.type == "complex") {
        mdouble tmp;
        while (num_entries_read < nnz && !ifile.eof()) {
            ifile >> I_[num_entries_read];
            ifile >> J_[num_entries_read];
            ifile >> coo_values_[num_entries_read] >> tmp;
            num_entries_read++;
        }
    } else {
        throw std::runtime_error("invalid MatrixMarket data type");
    }
    ifile.close();

    if (num_entries_read != nnz)
        throw std::runtime_error("read nnz not equal to decalred nnz " + std::to_string(num_entries_read));

    for (mint n = 0; n < nnz; n++) {
        I_[n] -= 1;
        J_[n] -= 1;
    }

    if (banner.symmetry != "general") {
        mint non_diagonals = 0;
        for (mint n = 0; n < nnz; n++)
            if (expect_true(I_[n] != J_[n])) non_diagonals++;
        const mint new_nnz = nnz + non_diagonals;

        mint *new_I = new mint[new_nnz];
        mint *new_J = new mint[new_nnz];
        mdouble *new_coo_values = new mdouble[new_nnz];

        if (banner.symmetry == "symmetric") {
            mint cnt = 0;
            for (mint n = 0; n < nnz; n++) {
                new_I[cnt] = I_[n];
                new_J[cnt] = J_[n];
                new_coo_values[cnt] = coo_values_[n];
                cnt++;
                if (I_[n] != J_[n]) {
                    new_I[cnt] = J_[n];
                    new_J[cnt] = I_[n];
                    new_coo_values[cnt] = coo_values_[n];
                    cnt++;
                }
            }
            assert(new_nnz == cnt && "something wrong: new_nnz != cnt");
        } else if (banner.symmetry == "skew-symmetric") {
            mint cnt = 0;
            for (mint n = 0; n < nnz; n++) {
                new_I[cnt] = I_[n];
                new_J[cnt] = J_[n];
                new_coo_values[cnt] = coo_values_[n];
                cnt++;
                if (I_[n] != J_[n]) {
                    new_I[cnt] = J_[n];
                    new_J[cnt] = I_[n];
                    new_coo_values[cnt] = -coo_values_[n];
                    cnt++;
                }
            }
            assert(new_nnz == cnt && "something wrong: new_nnz != cnt");
        } else if (banner.symmetry == "hermitian") {
            throw std::runtime_error("MatrixMarket I/O does not currently support hermitian matrices");
        }

        nnz = new_nnz;
        delete[] I_;
        delete[] J_;
        delete[] coo_values_;
        I_ = new_I;
        J_ = new_J;
        coo_values_ = new_coo_values;
    }

    Pair<long, mdouble> *p = new Pair<long, mdouble>[nnz];
    for (mint i = 0; i < nnz; i++) {
        p[i].ind = static_cast<long int>(N) * I_[i] + J_[i];
        p[i].val = coo_values_[i];
    }
    std::sort(p, p + nnz);
    for (mint i = 0; i < nnz; i++) {
        I_[i] = p[i].ind / N;
        J_[i] = p[i].ind % N;
        coo_values_[i] = p[i].val;
    }
    delete[] p;

    rpt = new mint[M + 1];
    std::memset(rpt, 0, (M + 1) * sizeof(mint));
    for (mint i = 0; i < nnz; i++) {
        rpt[I_[i] + 1]++;
    }
    for (mint i = 1; i <= M; i++) {
        rpt[i] += rpt[i - 1];
    }
    delete[] I_;
    col = J_;
    val = coo_values_;

    assert(rpt[0] == 0 && "first row_pointer != 0");
    for (mint i = 0; i < M; i++) {
        if (expect_true(rpt[i] <= rpt[i + 1] && rpt[i] <= nnz)) {
            for (mint j = rpt[i]; j < rpt[i + 1] - 1; j++) {
                if (expect_true(col[j] < col[j + 1])) {
                } else {
                    std::printf("row %d, col_index %d, index %d\n", i, col[j], j);
                    throw std::runtime_error("csr col_index not in assending order");
                }
            }
            for (mint j = rpt[i]; j < rpt[i + 1]; j++) {
                if (expect_true(col[j] < N && col[j] >= 0)) {
                } else {
                    std::printf("row %d, col_index %d, index %d\n", i, col[j], j);
                    throw std::runtime_error("csr col_index out of range");
                }
            }
        } else {
            std::printf("i %d  row_pointer[i] %d row_pointer[i+1] %d\n", i, rpt[i], rpt[i + 1]);
            throw std::runtime_error("csr row_pointer wrong");
        }
    }
    assert(rpt[M] == nnz && "last row_pointer != nnz_");
}

}  // namespace opsparse
