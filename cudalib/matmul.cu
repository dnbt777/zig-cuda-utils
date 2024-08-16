#include <cuda_runtime.h>
#include <stdio.h>

#define N 2 // Size of the matrices (N x N)

__global__ void matmulKernel(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

extern "C" void matmul(float *A, float *B, float *C) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

extern "C" int cuda_device_check() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable device found.\n");
        return -1; // Exit if no device is found
    }

    // Optionally, you can print device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Get properties of the first device
    printf("Using device: %s\n", deviceProp.name);

    // Example matrices
    float A[N * N] = {1, 2, 3, 4};
    float B[N * N] = {5, 6, 7, 8};
    float C[N * N] = {0};

    // Call the matrix multiplication function
    matmul(A, B, C);

    // Print the result
    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }
    return 0;
}

