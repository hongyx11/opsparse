#include "opsparse/core_algo/spgemm/spgemm_phases.cuh"

#include "opsparse/core_algo/spgemm/kernels/binning.cuh"
#include "opsparse/core_algo/spgemm/kernels/numeric.cuh"
#include "opsparse/core_algo/spgemm/kernels/setup.cuh"
#include "opsparse/core_algo/spgemm/kernels/symbolic.cuh"
#include "opsparse/numeric/compute_flop.hpp"
#include "opsparse/utils/config.hpp"

namespace opsparse {

void h_compute_flop(const CSR &A, const CSR &B, CSR &C, SpgemmMeta & /*meta*/)
{
    const mint BS = 1024;
    const mint GS = div_up(A.M, BS);
    k_compute_flop<<<GS, BS>>>(A.d_rpt, A.d_col, B.d_rpt, C.M, C.d_rpt, C.d_rpt + C.M);
}

void h_setup(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta, SpgemmTimings & /*timing*/)
{
    meta.allocate_rpt(C);
    cudaMemset(C.d_rpt + C.M, 0, sizeof(mint));
    h_compute_flop(A, B, C, meta);
    meta.allocate(C);
    OPSPARSE_CHECK_CUDA(cudaMemcpy(meta.max_row_nnz, C.d_rpt + C.M, sizeof(mint), cudaMemcpyDeviceToHost));
}

void h_symbolic_binning(CSR &C, SpgemmMeta &meta)
{
    meta.memset_all(0);
    const mint BS = 1024;
    const mint GS = div_up(C.M, BS);
    if (*meta.max_row_nnz <= 26) {
        k_binning_small<<<GS, BS>>>(meta.d_bins, C.M);
        meta.bin_size[0] = C.M;
        for (int i = 1; i < kNumBin; i++) meta.bin_size[i] = 0;
        meta.bin_offset[0] = 0;
        for (int i = 1; i < kNumBin; i++) meta.bin_offset[i] = C.M;
    } else {
        k_symbolic_binning<<<GS, BS, 0, meta.stream[0]>>>(C.d_rpt, C.M, meta.d_bin_size);
        meta.d2h_bin_size(0);
        meta.memset_bin_size(0);
        meta.bin_offset[0] = 0;
        for (int i = 0; i < kNumBin - 1; i++) {
            meta.bin_offset[i + 1] = meta.bin_offset[i] + meta.bin_size[i];
        }
        meta.h2d_bin_offset(0);
        k_symbolic_binning2<<<GS, BS, 0, meta.stream[0]>>>(C.d_rpt, C.M, meta.d_bins, meta.d_bin_size, meta.d_bin_offset);
    }
}

void h_symbolic(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta)
{
    if (meta.bin_size[5]) {
        k_symbolic_shared_hash_tb<8192><<<meta.bin_size[5], 1024, 0, meta.stream[5]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[5], C.d_rpt);
    }

    mint *d_fail_bins = nullptr;
    mint *d_fail_bin_size = nullptr;
    mint fail_bin_size = 0;
    if (meta.bin_size[7]) {
        if (meta.bin_size[7] + 1 <= static_cast<mint>(meta.cub_storage_size / sizeof(mint))) {
            d_fail_bins = meta.d_cub_storage;
            d_fail_bin_size = meta.d_cub_storage + meta.bin_size[7];
        } else {
            OPSPARSE_CHECK_CUDA(cudaMalloc(&d_fail_bins, (meta.bin_size[7] + 1) * sizeof(mint)));
            d_fail_bin_size = d_fail_bins + meta.bin_size[7];
        }
        OPSPARSE_CHECK_CUDA(cudaMemsetAsync(d_fail_bin_size, 0, sizeof(mint), meta.stream[7]));
        OPSPARSE_CHECK_CUDA(
            cudaFuncSetAttribute(k_symbolic_max_shared_hash_tb_with_fail, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
        k_symbolic_max_shared_hash_tb_with_fail<<<meta.bin_size[7], 1024, 98304, meta.stream[7]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[7], d_fail_bins, d_fail_bin_size, C.d_rpt);
    }
    if (meta.bin_size[6]) {
        k_symbolic_large_shared_hash_tb<<<meta.bin_size[6], 1024, 0, meta.stream[6]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[6], C.d_rpt);
    }
    if (meta.bin_size[0]) {
        const mint BS = kPwarpRows * kPwarp;
        const mint GS = div_up(meta.bin_size[0], kPwarpRows);
        k_symbolic_shared_hash_pwarp<<<GS, BS, 0, meta.stream[0]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[0], meta.bin_size[0], C.d_rpt);
    }

    if (meta.bin_size[7]) {
        OPSPARSE_CHECK_CUDA(
            cudaMemcpyAsync(&fail_bin_size, d_fail_bin_size, sizeof(mint), cudaMemcpyDeviceToHost, meta.stream[7]));
        OPSPARSE_CHECK_CUDA(cudaStreamSynchronize(meta.stream[7]));
        if (fail_bin_size) {
            const mint max_tsize = *meta.max_row_nnz * kSymbolicScaleLarge;
            meta.global_mem_pool_size = fail_bin_size * max_tsize * sizeof(mint);
            OPSPARSE_CHECK_CUDA(cudaMalloc(&meta.d_global_mem_pool, meta.global_mem_pool_size));
            meta.global_mem_pool_malloced = true;
            k_symbolic_global_hash_tb<<<fail_bin_size, 1024, 0, meta.stream[7]>>>(
                A.d_rpt, A.d_col, B.d_rpt, B.d_col, d_fail_bins, C.d_rpt, meta.d_global_mem_pool, max_tsize);
        }
    }

    if (meta.bin_size[4]) {
        k_symbolic_shared_hash_tb<4096><<<meta.bin_size[4], 512, 0, meta.stream[4]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[4], C.d_rpt);
    }
    if (meta.bin_size[3]) {
        k_symbolic_shared_hash_tb<2048><<<meta.bin_size[3], 256, 0, meta.stream[3]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[3], C.d_rpt);
    }
    if (meta.bin_size[2]) {
        k_symbolic_shared_hash_tb<1024><<<meta.bin_size[2], 128, 0, meta.stream[2]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[2], C.d_rpt);
    }
    if (meta.bin_size[1]) {
        k_symbolic_shared_hash_tb<512><<<meta.bin_size[1], 64, 0, meta.stream[1]>>>(
            A.d_rpt, A.d_col, B.d_rpt, B.d_col, meta.d_bins + meta.bin_offset[1], C.d_rpt);
    }

    if (meta.bin_size[7] && meta.bin_size[7] + 1 > static_cast<mint>(meta.cub_storage_size / sizeof(mint))) {
        OPSPARSE_CHECK_CUDA(cudaFree(d_fail_bins));
    }
}

void h_numeric_binning(CSR &C, SpgemmMeta &meta)
{
    meta.memset_all(0);
    const mint BS = 1024;
    const mint GS = div_up(C.M, BS);
    k_numeric_binning<<<GS, BS, 0, meta.stream[0]>>>(C.d_rpt, C.M, meta.d_bin_size, meta.d_total_nnz, meta.d_max_row_nnz);
    meta.d2h_all(0);
    OPSPARSE_CHECK_CUDA(cudaStreamSynchronize(meta.stream[0]));
    if (*meta.max_row_nnz <= 16) {
        k_binning_small<<<GS, BS>>>(meta.d_bins, C.M);
        meta.bin_size[0] = C.M;
        for (int i = 1; i < kNumBin; i++) meta.bin_size[i] = 0;
        meta.bin_offset[0] = 0;
        for (int i = 1; i < kNumBin; i++) meta.bin_offset[i] = C.M;
    } else {
        meta.memset_bin_size(0);
        meta.bin_offset[0] = 0;
        for (int i = 0; i < kNumBin - 1; i++) {
            meta.bin_offset[i + 1] = meta.bin_offset[i] + meta.bin_size[i];
        }
        meta.h2d_bin_offset(0);
        k_numeric_binning2<<<GS, BS, 0, meta.stream[0]>>>(C.d_rpt, C.M, meta.d_bins, meta.d_bin_size, meta.d_bin_offset);
    }
}

void h_numeric_full_occu(const CSR &A, const CSR &B, CSR &C, SpgemmMeta &meta)
{
    if (meta.bin_size[6]) {
        OPSPARSE_CHECK_CUDA(
            cudaFuncSetAttribute(k_numeric_max_shared_hash_tb_half_occu, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
        k_numeric_max_shared_hash_tb_half_occu<<<meta.bin_size[6], 1024, 98304, meta.stream[6]>>>(
            A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val, meta.d_bins + meta.bin_offset[6], C.d_rpt, C.d_col, C.d_val);
    }

    if (meta.bin_size[7]) {
        const mint max_tsize = *meta.max_row_nnz * kNumericScaleLarge;
        const std::size_t global_size = meta.bin_size[7] * max_tsize * (sizeof(mint) + sizeof(mdouble));
        if (meta.global_mem_pool_malloced) {
            if (global_size > meta.global_mem_pool_size) {
                OPSPARSE_CHECK_CUDA(cudaFree(meta.d_global_mem_pool));
                OPSPARSE_CHECK_CUDA(cudaMalloc(&meta.d_global_mem_pool, global_size));
            }
        } else {
            OPSPARSE_CHECK_CUDA(cudaMalloc(&meta.d_global_mem_pool, global_size));
            meta.global_mem_pool_size = global_size;
            meta.global_mem_pool_malloced = true;
        }
        k_numeric_global_hash_tb_full_occu<<<meta.bin_size[7], 1024, 0, meta.stream[7]>>>(
            A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val, meta.d_bins + meta.bin_offset[7], max_tsize,
            meta.d_global_mem_pool, C.d_rpt, C.d_col, C.d_val);
    }

    if (meta.bin_size[5]) {
        k_numeric_shared_hash_tb_full_occu<4096, 1024><<<meta.bin_size[5], 1024, 0, meta.stream[5]>>>(
            A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val, meta.d_bins + meta.bin_offset[5], C.d_rpt, C.d_col, C.d_val);
    }
    if (meta.bin_size[0]) {
        const mint BS = kNumericPwarpRows * kNumericPwarp;
        const mint GS = div_up(meta.bin_size[0], kNumericPwarpRows);
        k_numeric_shared_hash_pwarp<<<GS, BS, 0, meta.stream[0]>>>(
            A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val, meta.d_bins + meta.bin_offset[0], meta.bin_size[0], C.d_rpt,
            C.d_col, C.d_val);
    }
    if (meta.bin_size[4]) {
        k_numeric_shared_hash_tb_full_occu<2048, 512><<<meta.bin_size[4], 512, 0, meta.stream[4]>>>(
            A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val, meta.d_bins + meta.bin_offset[4], C.d_rpt, C.d_col, C.d_val);
    }
    if (meta.bin_size[3]) {
        k_numeric_shared_hash_tb_full_occu<1024, 256><<<meta.bin_size[3], 256, 0, meta.stream[3]>>>(
            A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val, meta.d_bins + meta.bin_offset[3], C.d_rpt, C.d_col, C.d_val);
    }
    if (meta.bin_size[2]) {
        k_numeric_shared_hash_tb_full_occu<512, 128><<<meta.bin_size[2], 128, 0, meta.stream[2]>>>(
            A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val, meta.d_bins + meta.bin_offset[2], C.d_rpt, C.d_col, C.d_val);
    }
    if (meta.bin_size[1]) {
        k_numeric_shared_hash_tb_full_occu<256, 64><<<meta.bin_size[1], 64, 0, meta.stream[1]>>>(
            A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val, meta.d_bins + meta.bin_offset[1], C.d_rpt, C.d_col, C.d_val);
    }

    if (meta.global_mem_pool_malloced) {
        OPSPARSE_CHECK_CUDA(cudaFree(meta.d_global_mem_pool));
    }
}

}  // namespace opsparse
