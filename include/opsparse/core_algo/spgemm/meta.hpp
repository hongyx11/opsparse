#ifndef OPSPARSE_CORE_ALGO_SPGEMM_META_HPP_
#define OPSPARSE_CORE_ALGO_SPGEMM_META_HPP_

#include <cstddef>

#include "opsparse/system/cuda_common.hpp"
#include "opsparse/utils/config.hpp"

namespace opsparse {

class CSR;

// Per-SpGEMM workspace: bin metadata, cub scratch, and lazily-allocated
// global memory pool for the large-hash fallback path.
class SpgemmMeta {
   public:
    // combined device / host memory (single allocation each)
    mint *d_combined_mem = nullptr;
    mint *combined_mem = nullptr;

    mint M = 0;
    mint N = 0;

    mint *d_bins = nullptr;          // size M
    mint *d_bin_size = nullptr;      // size kNumBin
    mint *d_bin_offset = nullptr;    // size kNumBin
    mint *d_max_row_nnz = nullptr;   // size 1
    mint *d_total_nnz = nullptr;     // size 1
    mint *d_cub_storage = nullptr;   // variable

    mint *bin_size = nullptr;        // size kNumBin
    mint *bin_offset = nullptr;      // size kNumBin
    mint *max_row_nnz = nullptr;     // size 1
    mint *total_nnz = nullptr;       // size 1

    std::size_t cub_storage_size = 0;
    cudaStream_t *stream = nullptr;

    // lazily allocated at runtime for the global-hash fallback
    mint *d_global_mem_pool = nullptr;
    std::size_t global_mem_pool_size = 0;
    bool global_mem_pool_malloced = false;

    SpgemmMeta() = default;
    SpgemmMeta(const SpgemmMeta &) = delete;
    SpgemmMeta &operator=(const SpgemmMeta &) = delete;
    explicit SpgemmMeta(CSR &C);

    void allocate_rpt(CSR &C);
    void allocate(CSR &C);
    void release();

    void memset_bin_size(mint stream_idx);
    void memset_all(mint stream_idx);
    void d2h_bin_size(mint stream_idx);
    void d2h_all(mint stream_idx);
    void h2d_bin_offset(mint stream_idx);

    ~SpgemmMeta();
};

}  // namespace opsparse

#endif  // OPSPARSE_CORE_ALGO_SPGEMM_META_HPP_
