#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addVector(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    int numElements = 1000; 
    int threadsPerBlock = 512; 
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    float *d_A = NULL;  cudaMalloc((void **)&d_A, size);
    float *d_B = NULL;  cudaMalloc((void **)&d_B, size);
    float *d_C = NULL;  cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    clock_t start_time = clock();

    addVector<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    clock_t stop_time = clock();
    int total_time = (int)(stop_time - start_time);
    fprintf("TOTAL TIME: %d", total_time);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}