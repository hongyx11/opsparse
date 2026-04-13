#include "opsparse/core_algo/spgemm/cusparse_spgemm.hpp"

#include <cusparse.h>

#include "opsparse/system/cuda_common.hpp"

namespace opsparse {

namespace {

void cusparse_spgemm_inner(int *d_row_ptr_A, int *d_col_idx_A, double *d_csr_values_A, int *d_row_ptr_B, int *d_col_idx_B,
                           double *d_csr_values_B, int **d_row_ptr_C, int **d_col_idx_C, double **d_csr_values_C, int M,
                           int K, int N, int nnz_A, int nnz_B, int *nnz_C)
{
    OPSPARSE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(d_row_ptr_C), (M + 1) * sizeof(int)));

    cusparseHandle_t handle;
    OPSPARSE_CHECK_CUSPARSE(cusparseCreate(&handle), "create cusparse handle");
    cusparseSpMatDescr_t matA, matB, matC;
    void *dBuffer1 = nullptr;
    void *dBuffer2 = nullptr;
    std::size_t bufferSize1 = 0;
    std::size_t bufferSize2 = 0;
    OPSPARSE_CHECK_CUSPARSE(cusparseCreateCsr(&matA, M, K, nnz_A, d_row_ptr_A, d_col_idx_A, d_csr_values_A,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                            "create matA");
    OPSPARSE_CHECK_CUSPARSE(cusparseCreateCsr(&matB, K, N, nnz_B, d_row_ptr_B, d_col_idx_B, d_csr_values_B,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                            "create matB");
    OPSPARSE_CHECK_CUSPARSE(cusparseCreateCsr(&matC, M, N, 0, nullptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                            "create matC");
    cusparseSpGEMMDescr_t spgemmDescr;
    OPSPARSE_CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDescr), "create spgemm descr");
    double alpha = 1.0;
    double beta = 0.0;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_64F;

    OPSPARSE_CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
                                                          CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1, nullptr),
                            "first work estimation");
    OPSPARSE_CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    OPSPARSE_CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
                                                          CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1, dBuffer1),
                            "second work estimation");
    OPSPARSE_CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
                                                    CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2, nullptr),
                            "first compute");

    OPSPARSE_CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2));
    OPSPARSE_CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
                                                    CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2, dBuffer2),
                            "second compute");

    int64_t M_C, N_C, nnz_C_64I;
    OPSPARSE_CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &M_C, &N_C, &nnz_C_64I));
    *nnz_C = static_cast<int>(nnz_C_64I);
    OPSPARSE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(d_col_idx_C), *nnz_C * sizeof(int)));
    OPSPARSE_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(d_csr_values_C), *nnz_C * sizeof(double)));
    OPSPARSE_CHECK_CUSPARSE(cusparseCsrSetPointers(matC, *d_row_ptr_C, *d_col_idx_C, *d_csr_values_C));

    OPSPARSE_CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
                                                 CUSPARSE_SPGEMM_DEFAULT, spgemmDescr),
                            "spgemm copy");
    OPSPARSE_CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDescr));
    OPSPARSE_CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    OPSPARSE_CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    OPSPARSE_CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    OPSPARSE_CHECK_CUSPARSE(cusparseDestroy(handle));

    OPSPARSE_CHECK_CUDA(cudaFree(dBuffer1));
    OPSPARSE_CHECK_CUDA(cudaFree(dBuffer2));

    OPSPARSE_CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace

void cusparse_spgemm(const CSR &A, const CSR &B, CSR &C)
{
    int tmp_nnz;
    cusparse_spgemm_inner(A.d_rpt, A.d_col, A.d_val, B.d_rpt, B.d_col, B.d_val, &C.d_rpt, &C.d_col, &C.d_val, A.M, A.N, B.N,
                          A.nnz, B.nnz, &tmp_nnz);
    C.M = A.M;
    C.N = B.N;
    C.nnz = tmp_nnz;
}

}  // namespace opsparse
