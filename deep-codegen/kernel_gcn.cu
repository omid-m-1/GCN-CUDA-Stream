#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// SpMMv Kernel
__global__ void spmmv_kernel(float* input, float* output, int* rowPtr, int* colInd, float* values, int* degrees, int F_in, int V) {
    // Outpuy: node v - feature f
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < V && f < F_in) {
        int row_start = rowPtr[v];
        int row_end = rowPtr[v + 1];
        float result = 0.0f;
        for (int i = row_start; i < row_end; ++i) {
            int col = colInd[i];
            float val = values[i];
            result += val * input[col * F_in + f];
        }
        output[v * F_in + f] = result / degrees[v];
    }
}

// SpMMv Function
void spmmv(array2d_t<float>& input, array2d_t<float>& output, array1d_t<int>& rowPtr, array1d_t<int>& colInd, array1d_t<float>& values, array1d_t<int>& degrees, int V, int F_in) {
    // Dense input and output
    float* ds_in = input.data_ptr;
    float* out = output.data_ptr;
    // Sparse input in csr
    int* row = rowPtr.data_ptr;
    int* col = colInd.data_ptr;
    float* val = values.data_ptr;
    int* d = degrees.data_ptr;
    // load kernel
    dim3 blockSize(32, 4);
    dim3 gridSize((F_in + blockSize.x -1) / blockSize.x, (V + blockSize.y - 1) / blockSize.y);
    spmmv_kernel<<<gridSize, blockSize>>>(ds_in, out, row, col, val, d, F_in, V);
}
