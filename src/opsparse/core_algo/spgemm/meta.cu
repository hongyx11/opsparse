#include "opsparse/core_algo/spgemm/meta.hpp"

#include <cassert>
#include <cstdlib>
#include <cub/cub.cuh>

#include "opsparse/core_ds/csr.hpp"
#include "opsparse/utils/config.hpp"

namespace opsparse {

SpgemmMeta::SpgemmMeta(CSR &C) { allocate_rpt(C); }

void SpgemmMeta::allocate_rpt(CSR &C) { OPSPARSE_CHECK_CUDA(cudaMalloc(&C.d_rpt, (C.M + 1) * sizeof(mint))); }

void SpgemmMeta::allocate(CSR &C)
{
    M = C.M;
    N = C.N;
    stream = new cudaStream_t[kNumBin];
    for (int i = 0; i < kNumBin; i++) {
        OPSPARSE_CHECK_CUDA(cudaStreamCreate(stream + i));
    }

    cub::DeviceScan::ExclusiveSum(nullptr, cub_storage_size, C.d_rpt, C.d_rpt, M + 1);

    const mint d_combined_size = M + 2 * kNumBin + 2 + cub_storage_size / sizeof(mint);
    OPSPARSE_CHECK_CUDA(cudaMalloc(&d_combined_mem, d_combined_size * sizeof(mint)));
    const mint combined_size = 2 * kNumBin + 2;
    combined_mem = static_cast<mint *>(std::malloc(combined_size * sizeof(mint)));
    assert(combined_mem != nullptr);

    d_bins = d_combined_mem;
    d_bin_size = d_combined_mem + M;
    d_max_row_nnz = d_bin_size + kNumBin;
    d_total_nnz = d_bin_size + kNumBin + 1;
    d_bin_offset = d_total_nnz + 1;
    d_cub_storage = d_bin_offset + 1;

    bin_size = combined_mem;
    max_row_nnz = bin_size + kNumBin;
    total_nnz = bin_size + kNumBin + 1;
    bin_offset = bin_size + kNumBin + 2;

    d_global_mem_pool = nullptr;
    global_mem_pool_size = 0;
    global_mem_pool_malloced = false;
}

void SpgemmMeta::release()
{
    if (d_combined_mem) {
        cudaFree(d_combined_mem);
        d_combined_mem = nullptr;
    }
    if (stream != nullptr) {
        for (int i = 0; i < kNumBin; i++) {
            cudaStreamDestroy(stream[i]);
        }
        delete[] stream;
        stream = nullptr;
    }
    std::free(combined_mem);
    combined_mem = nullptr;
}

SpgemmMeta::~SpgemmMeta() { release(); }

void SpgemmMeta::memset_all(mint stream_idx)
{
    OPSPARSE_CHECK_CUDA(cudaMemsetAsync(d_bin_size, 0, (kNumBin + 2) * sizeof(mint), stream[stream_idx]));
}

void SpgemmMeta::memset_bin_size(mint stream_idx)
{
    OPSPARSE_CHECK_CUDA(cudaMemsetAsync(d_bin_size, 0, kNumBin * sizeof(mint), stream[stream_idx]));
}

void SpgemmMeta::d2h_all(mint stream_idx)
{
    OPSPARSE_CHECK_CUDA(
        cudaMemcpyAsync(bin_size, d_bin_size, (kNumBin + 2) * sizeof(mint), cudaMemcpyDeviceToHost, stream[stream_idx]));
}

void SpgemmMeta::d2h_bin_size(mint stream_idx)
{
    OPSPARSE_CHECK_CUDA(
        cudaMemcpyAsync(bin_size, d_bin_size, kNumBin * sizeof(mint), cudaMemcpyDeviceToHost, stream[stream_idx]));
}

void SpgemmMeta::h2d_bin_offset(mint stream_idx)
{
    OPSPARSE_CHECK_CUDA(
        cudaMemcpyAsync(d_bin_offset, bin_offset, kNumBin * sizeof(mint), cudaMemcpyHostToDevice, stream[stream_idx]));
}

}  // namespace opsparse
