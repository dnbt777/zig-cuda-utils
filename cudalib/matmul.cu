#include <cuda_runtime.h>
#include <stdio.h>

#define N 2048 // Size of the matrices (N x N)

__global__ void matmulKernel(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

extern "C" void matmul(float *A, float *B, float *C) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate device memory and check for errors
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for A: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Copy matrices from host to device
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying A to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying B to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the kernel
    matmulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Copy result from device to host
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying C to host: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
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

    // Print device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Get properties of the first device
    printf("Using device: %s\n", deviceProp.name);

    return 0;
}

