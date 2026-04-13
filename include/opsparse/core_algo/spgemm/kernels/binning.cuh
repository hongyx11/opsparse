#ifndef OPSPARSE_CORE_ALGO_SPGEMM_KERNELS_BINNING_CUH_
#define OPSPARSE_CORE_ALGO_SPGEMM_KERNELS_BINNING_CUH_

#include <climits>

#include "opsparse/system/cuda_common.hpp"
#include "opsparse/utils/config.hpp"

namespace opsparse {

static __global__ void __launch_bounds__(1024, 2) k_symbolic_binning(mint *d_row_flop, int M, mint *d_bin_size)
{
    __shared__ mint shared_bin_size[kNumBin];
    if (threadIdx.x < kNumBin) {
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();

    mint i = threadIdx.x + blockIdx.x * blockDim.x;
    mint row_nnz, j;
    mint range[kNumBin] = {26, 426, 853, 1706, 3413, 6826, 10240, INT_MAX};  // 1.2x
    if (i < M) {
        row_nnz = d_row_flop[i];
        for (j = 0; j < kNumBin; j++) {
            if (row_nnz <= range[j]) {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
before_end:
    __syncthreads();
    if (threadIdx.x < kNumBin) {
        atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
    }
}

static __global__ void __launch_bounds__(1024, 2)
    k_symbolic_binning2(mint *__restrict__ d_row_flop, int M, mint *__restrict__ d_bins, mint *__restrict__ d_bin_size,
                        mint *__restrict__ d_bin_offset)
{
    __shared__ mint shared_bin_size[kNumBin];
    __shared__ mint shared_bin_offset[kNumBin];
    if (threadIdx.x < kNumBin) {
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();

    mint i = threadIdx.x + blockIdx.x * blockDim.x;
    mint row_nnz, j;
    mint range[kNumBin] = {26, 426, 853, 1706, 3413, 6826, 10240, INT_MAX};  // 1.2x
    if (i < M) {
        row_nnz = d_row_flop[i];
        for (j = 0; j < kNumBin; j++) {
            if (row_nnz <= range[j]) {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
before_end:

    __syncthreads();
    if (threadIdx.x < kNumBin) {
        shared_bin_offset[threadIdx.x] = atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
        shared_bin_offset[threadIdx.x] += d_bin_offset[threadIdx.x];
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();

    mint index;
    if (i < M) {
        for (j = 0; j < kNumBin; j++) {
            if (row_nnz <= range[j]) {
                index = atomicAdd(shared_bin_size + j, 1);
                d_bins[shared_bin_offset[j] + index] = i;
                return;
            }
        }
    }
}

static __global__ void k_binning_small(mint *d_bins, mint M)
{
    mint i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= M) return;
    d_bins[i] = i;
}

static __global__ void __launch_bounds__(1024, 2)
    k_numeric_binning(mint *__restrict__ d_row_nnz, int M, mint *__restrict__ d_bin_size, mint *__restrict__ d_total_nnz,
                      mint *__restrict__ d_max_row_nnz)
{
    __shared__ mint shared_bin_size[kNumBin];
    __shared__ mint shared_local_nnz[1];
    __shared__ mint shared_max_row_nnz[1];
    if (threadIdx.x < kNumBin) {
        shared_bin_size[threadIdx.x] = 0;
    }
    if (threadIdx.x == 32) {
        shared_local_nnz[0] = 0;
        shared_max_row_nnz[0] = 0;
    }
    __syncthreads();
    mint range[kNumBin] = {16, 128, 256, 512, 1024, 2048, 4095, INT_MAX};  // 2x
    mint i = threadIdx.x + blockIdx.x * blockDim.x;
    mint row_nnz, j;
    if (i < M) {
        row_nnz = d_row_nnz[i];
        atomicAdd(shared_local_nnz, row_nnz);
        atomicMax(shared_max_row_nnz, row_nnz);
        for (j = 0; j < kNumBin; j++) {
            if (row_nnz <= range[j]) {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
before_end:

    __syncthreads();
    if (threadIdx.x < kNumBin) {
        atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
    }
    if (threadIdx.x == 32) {
        atomicAdd(d_total_nnz, shared_local_nnz[0]);
    }
    if (threadIdx.x == 64) {
        atomicMax(d_max_row_nnz, shared_max_row_nnz[0]);
    }
}

static __global__ void __launch_bounds__(1024, 2)
    k_numeric_binning2(mint *__restrict__ d_row_nnz, int M, mint *__restrict__ d_bins, mint *__restrict__ d_bin_size,
                       mint *__restrict__ d_bin_offset)
{
    __shared__ mint shared_bin_size[kNumBin];
    __shared__ mint shared_bin_offset[kNumBin];
    if (threadIdx.x < kNumBin) {
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    mint range[kNumBin] = {16, 128, 256, 512, 1024, 2048, 4095, INT_MAX};  // 2x
    mint i = threadIdx.x + blockIdx.x * blockDim.x;
    mint row_nnz, j;
    if (i < M) {
        row_nnz = d_row_nnz[i];
        for (j = 0; j < kNumBin; j++) {
            if (row_nnz <= range[j]) {
                atomicAdd(shared_bin_size + j, 1);
                goto before_end;
            }
        }
    }
before_end:

    __syncthreads();
    if (threadIdx.x < kNumBin) {
        shared_bin_offset[threadIdx.x] = atomicAdd(d_bin_size + threadIdx.x, shared_bin_size[threadIdx.x]);
        shared_bin_offset[threadIdx.x] += d_bin_offset[threadIdx.x];
        shared_bin_size[threadIdx.x] = 0;
    }
    __syncthreads();
    mint index;
    if (i < M) {
        for (j = 0; j < kNumBin; j++) {
            if (row_nnz <= range[j]) {
                index = atomicAdd(shared_bin_size + j, 1);
                d_bins[shared_bin_offset[j] + index] = i;
                return;
            }
        }
    }
}

}  // namespace opsparse

#endif  // OPSPARSE_CORE_ALGO_SPGEMM_KERNELS_BINNING_CUH_
