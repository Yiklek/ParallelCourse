#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cugemm.h>

int cu_gemm_float(int m, int k, int n, float *A, float *B, float *C)
{
    return cu_gemm<false>(m, n, k, A, B, C);
}
template <typename T, typename S>
void allocate_memory(int m, int n, int k, T **A, T **B, S **C) {
    cudaMallocManaged(A, m * k * sizeof(T));
    cudaMallocManaged(B, k * n * sizeof(T));
    cudaMallocManaged(C, m * n * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
template <bool IsCudaMem, typename T, typename S>
int cu_gemm(int m, int n, int k, T *A, T *B, S *C){
    float alpha = 1.0;
    float beta = 0.0;
    T *ra, *rb; S *rc;
    if(!IsCudaMem){
    //   printf("====>>> m:%d n:%d k:%d\n", m, n, k);
      cudaMalloc(&ra, m * k * sizeof(T));
      cudaMalloc(&rb, k * n * sizeof(T));
      cudaMalloc(&rc, m * n * sizeof(S));
      cudaMemcpy(ra, A, m * k * sizeof(T), cudaMemcpyHostToDevice);
    //   for (int i = 0; i < m * k; ++i)
    //         printf("%.5f%c", A[i], " \n"[(i+1) % k == 0]);
      cudaMemcpy(rb, B, k * n * sizeof(T), cudaMemcpyHostToDevice);
    //   for (int i = 0; i < n * k; ++i)
    //         printf("%.5f%c", B[i], " \n"[(i+1) % k == 0]);
    } else {
      ra = A;
      rb = B;
      rc = C;
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaDeviceSynchronize();
    int ret = cublas_gemm_ex(handle,
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 n,
                                 m,
                                 k,
                                 rb,
                                 ra,
                                 rc,
                                 k,
                                 k,
                                 n,
                                 &alpha,
                                 &beta,
                                 static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
    cudaDeviceSynchronize();
    if(!IsCudaMem){
      cudaMemcpy(C, rc, m * n * sizeof(S), cudaMemcpyDeviceToHost);
      free_memory(ra, rb, rc);
    }
    return ret;
}
template <typename T, typename S>
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                   int m, int n, int k, T *A, T *B, S *C, int lda, int ldb, int ldc,
                   S *alpha, S *beta, int algo) {
    cudaDataType_t AType, BType, CType, ComputeType;
    if (std::is_same<T, float>::value) {
        AType = BType = CType = ComputeType = CUDA_R_32F;
    } else if (std::is_same<T, __half>::value) {
        AType = BType = CType = ComputeType = CUDA_R_16F;
    } else if (std::is_same<T, int8_t>::value) {
        AType = BType = CUDA_R_8I;
        CType = ComputeType = CUDA_R_32I;
    } else {
        printf("Not supported data type.");
        return -1;
    }
    cublasStatus_t status;
    status = cublasGemmEx(handle,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          alpha,
                          A,
                          AType,
                          lda,
                          B,
                          BType,
                          ldb,
                          beta,
                          C,
                          CType,
                          ldc,
                          ComputeType,
                          static_cast<cublasGemmAlgo_t>(algo));
    
    if (status == CUBLAS_STATUS_SUCCESS)
        return 1;
    else
        return -1;
}

template <typename T, typename S>
void test_gemm(cublasHandle_t handle, int m, int n, int k, T *A, T *B, S *C,
               S *alpha, S *beta, int algo, int iteration) {
    float total_time = 0;
    for (int i = 0; i < iteration; ++i) {
        struct timeval start, end;
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);
        int success = cublas_gemm_ex(handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     n,
                                     m,
                                     k,
                                     B,
                                     A,
                                     C,
                                     n,
                                     k,
                                     n,
                                     alpha,
                                     beta,
                                     static_cast<cublasGemmAlgo_t>(algo));
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        cudaProfilerStop();
        if (success > 0 && i > 0)
            total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
    if (total_time > 0)
        printf("algo %d: %.3f ms\n", algo, total_time / (iteration - 1));
}
/*
int main() {
    int m = 2, n = 2, k = 2;
    printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);
    int start_algo = CUBLAS_GEMM_DEFAULT;
    int end_algo = CUBLAS_GEMM_ALGO23;
    int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;
    int iteration = 10;

    float *fA, *fB, *fC;
    float *hA = new float(m*k), *hB = new float(k*n), *hC = new float(m * n);
    float f_alpha = 1, f_beta = 0;
    allocate_memory(m, n, k, &fA, &fB, &fC);
    for (int i = 0; i < m * k; ++i) {
        fA[i] = 1;
        hA[i] = 1;
    } 
    for (int i = 0; i < k * n; ++i) {
        fB[i] = 1;
        hB[i] = 1;
    } 
    //int success = cu_gemm<true>(m, n, k, fA, fB, fC);
    int success = cu_gemm<false>(m, n, k, hA, hB, hC);
    printf("fp32: %d\n", success);
    for (int i = 0; i < m * n; ++i)
        printf("%.5f%c", hC[i], " \n"[(i+1) % n == 0]);
    free_memory(fA, fB, fC);
    return 0;
}
*/
