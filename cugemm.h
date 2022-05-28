#ifndef CU_GEMM_H
#define CU_GEMM_H
template <bool IsCudaMem = true, typename T, typename S>
int cu_gemm(int m, int n, int k, T *A, T *B, S *C);
 
extern "C" {
int cu_gemm_float(int m, int k, int n, float *A, float *B, float *C);
}
#endif
