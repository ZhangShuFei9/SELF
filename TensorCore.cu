#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

// 矩阵维度
#define M 1024
#define N 1024
#define K 1024

// Warp矩阵分块大小
#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 线程块配置
#define BLOCK_ROW_WARPS 4
#define BLOCK_COL_WARPS 4
#define BLOCK_SIZE (BLOCK_ROW_WARPS * BLOCK_COL_WARPS * WARP_SIZE)

// 每个线程块计算的矩阵分块大小
#define BLOCK_M (BLOCK_ROW_WARPS * WMMA_M)  // 64
#define BLOCK_N (BLOCK_COL_WARPS * WMMA_N)  // 64
#define BLOCK_K (WMMA_K)                    // 16

__global__ void wmma_matmul(half *A, half *B, float *C, 
                           int M, int N, int K) {
    // 动态共享内存用于存储矩阵分块
    extern __shared__ half shared_mem[];
    
    // 为A和B矩阵分块分配共享内存
    half *shared_a = shared_mem;
    half *shared_b = &shared_mem[BLOCK_M * BLOCK_K];
    
    // 计算线程块在输出矩阵中的位置
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    // 计算线程在warp中的位置
    int warp_row = threadIdx.y / 4;
    int warp_col = (threadIdx.x % 16) / 4;
    int lane_id = threadIdx.x % 32;
    
    // 声明wmma片段
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, 
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, 
                   half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, 
                   float> acc_frag[2];
    
    // 初始化累加器
    #pragma unroll
    for(int i=0; i<2; i++) {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }
    
    // 计算全局内存中的矩阵起始位置
    int a_row = block_row * BLOCK_M;
    int b_col = block_col * BLOCK_N;
    
    // 循环遍历K维度
    for (int k_step = 0; k_step < K; k_step += BLOCK_K) {
        // 协作加载A分块到共享内存
        int a_col = k_step;
        for (int i = threadIdx.y; i < BLOCK_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < BLOCK_K; j += blockDim.x) {
                int row = a_row + i;
                int col = a_col + j;
                if (row < M && col < K) {
                    shared_a[i * BLOCK_K + j] = A[row * K + col];
                } else {
                    shared_a[i * BLOCK_K + j] = __float2half(0.0f);
                }
            }
        }
        
        // 协作加载B分块到共享内存
        int b_row = k_step;
        for (int i = threadIdx.y; i < BLOCK_K; i += blockDim.y) {
            for (int j = threadIdx.x; j < BLOCK_N; j += blockDim.x) {
                int row = b_row + i;
                int col = b_col + j;
                if (row < K && col < N) {
                    shared_b[i * BLOCK_N + j] = B[row * N + col];
                } else {
                    shared_b[i * BLOCK_N + j] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // 从共享内存加载到片段并计算
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            // 加载A片段
            wmma::load_matrix_sync(a_frag, 
                &shared_a[warp_row * WMMA_M * BLOCK_K + kk], 
                BLOCK_K);
            
            // 加载B片段
            wmma::load_matrix_sync(b_frag, 
                &shared_b[kk * BLOCK_N + warp_col * WMMA_N], 
                BLOCK_N);
            
            // Tensor Core 计算
            wmma::mma_sync(acc_frag[0], a_frag, b_frag, acc_frag[0]);
        }
        
        __syncthreads();
    }
    
    // 存储结果到全局内存
    int c_row = a_row + warp_row * WMMA_M;
    int c_col = b_col + warp_col * WMMA_N;
    
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(&C[c_row * N + c_col], 
                               acc_frag[0], N, wmma::mem_row_major);
    }
}