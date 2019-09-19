#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matrixMultiply(const float *A, const float *B, float *C, int numElements)
{
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float counter = 0;

    if (ROW < numElements && COL < numElements) 
    {
        for (int i = 0; i < numElements; i++) 
        {
            counter += A[ROW * numElements + i] * B[i * numElements + COL];
        }
    }

    C[ROW * numElements + COL] = counter;
}

int main(void)
{
    int numElements = 1000; 
    int threadsPerBlock = 512; 
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    int matrixOrder = 4;

    for (int i = 0; i < matrixOrder; i++)
    {
        for (int j = 0; j < matrixOrder; j++)
        {
            h_A[(i * matrixOrder) + j] = rand()/(float)RAND_MAX;
            h_B[(i * matrixOrder) + j] = rand()/(float)RAND_MAX;
        }
    }

    float *d_A = NULL;  cudaMalloc((void **)&d_A, size);
    float *d_B = NULL;  cudaMalloc((void **)&d_B, size);
    float *d_C = NULL;  cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double memXFers=5*4*COLUMNS*ROWS;
    memXFers/=1024*1024*1024;

    printf("GPU: %f ms bandwidth %g GB/s",ms, memXFers/(ms/1000.0));
    printf("\n CPU : %g ms bandwidth %g GB/s",mtime, memXFers/(mtime/1000.0));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}